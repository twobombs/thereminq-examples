# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pyqrack import QrackSimulator
import math
import time

# --- Simulation Parameters ---
NUM_X_QUBITS = 6
NUM_Y_QUBITS = 6
NUM_POS_QUBITS = NUM_X_QUBITS + NUM_Y_QUBITS
NUM_COIN_QUBITS = 2
NUM_ANCILLA_QUBITS = 1
TOTAL_QUBITS = NUM_POS_QUBITS + NUM_COIN_QUBITS + NUM_ANCILLA_QUBITS

NUM_STEPS = 50
INTERPOLATION_FRAMES = 1 # Set to 1 for no extra interpolation between steps

INITIAL_X = 1 << (NUM_X_QUBITS // 2)
INITIAL_Y = 1 << (NUM_Y_QUBITS // 2)

BARRIER_X_COORD = (1 << NUM_X_QUBITS) - 8
BARRIER_PHASE = np.pi / 4

# --- Qubit Index Mapping ---
X_INDICES = list(range(NUM_X_QUBITS))
Y_INDICES = list(range(NUM_X_QUBITS, NUM_X_QUBITS + NUM_Y_QUBITS))
COIN_INDICES = list(range(NUM_POS_QUBITS, NUM_POS_QUBITS + NUM_COIN_QUBITS))
ANC_INDEX = NUM_POS_QUBITS + NUM_COIN_QUBITS

# --- Helper Functions ---
# (controlled_increment_qrack, controlled_decrement_qrack, apply_2d_barrier_qrack, walk_step_2d_qrack remain unchanged)
def controlled_increment_qrack(sim, target_indices, control_index, num_target_qubits):
    for i in range(num_target_qubits - 1):
        controls = [control_index] + target_indices[:i+1]
        sim.mcx(controls, target_indices[i+1])
    sim.mcx([control_index], target_indices[0])
    for i in range(num_target_qubits - 2, -1, -1):
        controls = [control_index] + target_indices[:i+1]
        sim.mcx(controls, target_indices[i+1])
        if i > 0:
             controls_inner = [control_index] + target_indices[:i]
             sim.mcx(controls_inner, target_indices[i])

def controlled_decrement_qrack(sim, target_indices, control_index, num_target_qubits):
    for i in range(num_target_qubits - 1):
         if i > 0:
             controls_inner = [control_index] + target_indices[:i]
             sim.mcx(controls_inner, target_indices[i])
         controls = [control_index] + target_indices[:i+1]
         sim.mcx(controls, target_indices[i+1])
    sim.mcx([control_index], target_indices[0])
    for i in range(num_target_qubits - 2, -1, -1):
        controls = [control_index] + target_indices[:i+1]
        sim.mcx(controls, target_indices[i+1])

def apply_2d_barrier_qrack(sim, x_indices, barrier_x_coord, barrier_phase, num_x_qubits):
    if barrier_x_coord < 0 or barrier_x_coord >= (1 << num_x_qubits): return
    barrier_bin = format(barrier_x_coord, f'0{num_x_qubits}b')[::-1]
    control_indices_for_phase = []
    for i in range(num_x_qubits):
        if barrier_bin[i] == '0': sim.x(x_indices[i])
        control_indices_for_phase.append(x_indices[i])
    if len(control_indices_for_phase) > 1:
        controls = control_indices_for_phase[:-1]; target = control_indices_for_phase[-1]
        sim.mcp(barrier_phase, controls, target)
    elif len(control_indices_for_phase) == 1:
        sim.p(barrier_phase, control_indices_for_phase[0])
    for i in range(num_x_qubits):
        if barrier_bin[i] == '0': sim.x(x_indices[i])

def walk_step_2d_qrack(sim, x_indices, y_indices, coin_indices, anc_index):
    for idx in coin_indices: sim.h(idx)
    for idx in coin_indices: sim.x(idx)
    sim.h(coin_indices[1]); sim.mcx([coin_indices[0]], coin_indices[1]); sim.h(coin_indices[1])
    for idx in coin_indices: sim.x(idx)
    c1, c0 = coin_indices[1], coin_indices[0]
    sim.x(c0); sim.x(c1); sim.mcx([c0, c1], anc_index); controlled_decrement_qrack(sim, y_indices, anc_index, NUM_Y_QUBITS); sim.mcx([c0, c1], anc_index); sim.x(c0); sim.x(c1)
    sim.x(c1); sim.mcx([c0, c1], anc_index); controlled_increment_qrack(sim, y_indices, anc_index, NUM_Y_QUBITS); sim.mcx([c0, c1], anc_index); sim.x(c1)
    sim.x(c0); sim.mcx([c0, c1], anc_index); controlled_decrement_qrack(sim, x_indices, anc_index, NUM_X_QUBITS); sim.mcx([c0, c1], anc_index); sim.x(c0)
    sim.mcx([c0, c1], anc_index); controlled_increment_qrack(sim, x_indices, anc_index, NUM_X_QUBITS); sim.mcx([c0, c1], anc_index)

# --- MODIFIED Probability Processing Function ---
# (This function expects a probability vector as input)
def process_qrack_probabilities(probs_vector, total_qubits, num_x, num_y):
    num_x_pos = 1 << num_x
    num_y_pos = 1 << num_y
    probabilities_2d = np.zeros((num_x_pos, num_y_pos))
    probs_vector = np.asarray(probs_vector)
    expected_len = 1 << total_qubits
    if len(probs_vector) != expected_len:
        # It's possible prob_all() might return fewer if states are guaranteed zero.
        # Handle carefully or assume full length for now.
        print(f"Warning: Probability vector length {len(probs_vector)} != expected {expected_len}.")
        # Adjust loop range if needed, or raise error depending on expected behavior
        # raise ValueError(...)

    x_mask = (1 << num_x) - 1
    y_mask = ((1 << num_y) - 1) << num_x
    # Use min to avoid index error if vector is unexpectedly short
    for i in range(min(len(probs_vector), expected_len)):
        x_index = i & x_mask
        y_index = (i & y_mask) >> num_x
        if 0 <= x_index < num_x_pos and 0 <= y_index < num_y_pos:
             probabilities_2d[x_index, y_index] += probs_vector[i]
    return probabilities_2d

# --- Initialize Simulator ---
print(f"Initializing QrackSimulator with {TOTAL_QUBITS} qubits...")
sim = QrackSimulator(TOTAL_QUBITS, isOpenCL=True)

# --- Set Initial State ---
print("Setting initial state...")
initial_x_bin = format(INITIAL_X, f'0{NUM_X_QUBITS}b')[::-1]
initial_y_bin = format(INITIAL_Y, f'0{NUM_Y_QUBITS}b')[::-1]
for i in range(NUM_X_QUBITS):
    if initial_x_bin[i] == '1': sim.x(X_INDICES[i])
for i in range(NUM_Y_QUBITS):
    if initial_y_bin[i] == '1': sim.x(Y_INDICES[i])
for idx in COIN_INDICES: sim.h(idx)

# --- Simulation ---
raw_probabilities_2d = []
max_prob = 0
num_x_positions = 1 << NUM_X_QUBITS
num_y_positions = 1 << NUM_Y_QUBITS

print(f"Starting simulation for {NUM_STEPS} steps...")
sim_start_time = time.time()

# Step 0
print("Processing Step 0...")
try:
    # probs_vector_raw = sim.get_probabilities() # Previous attempt
    probs_vector_raw = sim.prob_all() # <<< Potential Fix: Use prob_all()
    step_probs = process_qrack_probabilities(probs_vector_raw, TOTAL_QUBITS, NUM_X_QUBITS, NUM_Y_QUBITS)
    raw_probabilities_2d.append(step_probs)
    max_prob = max(max_prob, step_probs.max())
except AttributeError:
    print("ERROR: sim.prob_all() does not exist. Cannot proceed.")
    raise # Re-raise the error to stop execution
except Exception as e:
    print(f"ERROR during Step 0 probability processing: {e}")
    raise

# Main loop
for step in range(NUM_STEPS):
    print(f"Simulating Step {step + 1}/{NUM_STEPS}...")
    step_start_time = time.time()
    walk_step_2d_qrack(sim, X_INDICES, Y_INDICES, COIN_INDICES, ANC_INDEX)
    apply_2d_barrier_qrack(sim, X_INDICES, BARRIER_X_COORD, BARRIER_PHASE, NUM_X_QUBITS)

    try:
        # probs_vector_raw = sim.get_probabilities() # Previous attempt
        probs_vector_raw = sim.prob_all() # <<< Potential Fix: Use prob_all()
        step_probs = process_qrack_probabilities(probs_vector_raw, TOTAL_QUBITS, NUM_X_QUBITS, NUM_Y_QUBITS)
        raw_probabilities_2d.append(step_probs)
        max_prob = max(max_prob, step_probs.max())
    except AttributeError:
        print("ERROR: sim.prob_all() does not exist. Stopping simulation.")
        break
    except Exception as e:
        print(f"ERROR during Step {step+1} probability processing: {e}")
        break

    print(f"Step {step + 1} time: {time.time() - step_start_time:.2f} sec")

sim_end_time = time.time()
print(f"\nSimulation finished. Total time: {sim_end_time - sim_start_time:.2f} seconds")

# --- Plotting Code (remains the same) ---
if not raw_probabilities_2d:
    print("No probability data generated, skipping animation.")
else:
    print("Preparing frames for animation...")
    interpolated_probabilities_2d = list(raw_probabilities_2d)
    total_animation_frames = len(interpolated_probabilities_2d)
    print("Setting up 3D animation...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    x_coords = np.arange(num_x_positions)
    y_coords = np.arange(num_y_positions)
    X, Y = np.meshgrid(x_coords, y_coords)
    plot_max_prob = max(max_prob * 1.1, 1e-9)
    surf = ax.plot_surface(X, Y, interpolated_probabilities_2d[0].T, cmap='viridis', edgecolor='none', vmin=0, vmax=plot_max_prob)
    ax.set_xlabel("X Position"); ax.set_ylabel("Y Position"); ax.set_zlabel("Probability")
    ax.set_zlim(0, plot_max_prob)
    title = ax.set_title(f"2D Quantum Walk (Qrack) - Step 0 / {NUM_STEPS}")

    def update_3d(frame):
        ax.clear()
        surf = ax.plot_surface(X, Y, interpolated_probabilities_2d[frame].T, cmap='viridis', edgecolor='none', vmin=0, vmax=plot_max_prob)
        ax.set_xlabel("X Position"); ax.set_ylabel("Y Position"); ax.set_zlabel("Probability")
        ax.set_zlim(0, plot_max_prob)
        ax.set_title(f"2D Quantum Walk (Qrack) - Step {frame} / {NUM_STEPS}")
        return surf,

    ani = animation.FuncAnimation(fig, update_3d, frames=total_animation_frames, interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

    # Optional Save
    # print("Saving animation...")
    # try:
    #     ani.save('qrack_walk_2d_3d.mp4', writer='ffmpeg', fps=15, dpi=150)
    #     print("Animation saved successfully.")
    # except Exception as e:
    #     print(f"Error saving animation: {e}")
