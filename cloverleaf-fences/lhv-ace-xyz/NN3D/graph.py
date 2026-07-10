import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
DATA_FILE = "macroscopic_lattice_states.npy"
CONFIG_FILE = "lattice_config.json"
# ---------------------

def main():
    # 1. Dynamically load grid configuration
    print(f"Loading lattice configuration from {CONFIG_FILE}...")
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            grid_x = config.get("grid_x", 1)
            grid_y = config.get("grid_y", 1)
            grid_z = config.get("grid_z", 1)
            expected_patches = config.get("num_patches", grid_x * grid_y * grid_z)
    except FileNotFoundError:
        print(f"Error: {CONFIG_FILE} not found. Ensure you are running Engine Revision 44+.")
        sys.exit(1)

    print(f"Grid detected: {grid_x} x {grid_y} x {grid_z} ({expected_patches} patches)")

    # 2. Load the state history
    print(f"Loading state history from {DATA_FILE}...")
    try:
        # History shape: (steps, patches, 27_qubits, 3_axes)
        history = np.load(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run the hadron engine first.")
        sys.exit(1)

    num_steps = history.shape[0]
    num_patches = history.shape[1]
    
    if num_patches != expected_patches:
        print(f"Warning: Data file has {num_patches} patches, but config expected {expected_patches}.")

    print(f"Loaded {num_steps} steps across {num_patches} patches ({num_patches * 27} total qubits).")

    # 3. Reconstruct Global 3D Coordinates
    patch_coords = {}
    idx = 0
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                patch_coords[idx] = (x, y, z)
                idx += 1
                
    q_coords = {}
    lx, ly, lz = 3, 3, 3
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                q_idx = x * (ly * lz) + y * lz + z
                q_coords[q_idx] = (x, y, z)

    # Flatten coordinates into global space
    global_X = []
    global_Y = []
    global_Z = []
    
    for p in range(num_patches):
        px, py, pz = patch_coords.get(p, (0,0,0))
        for q in range(27):
            qx, qy, qz = q_coords[q]
            # Patches are 3 qubits wide, so global offset is patch_coord * 3
            global_X.append(px * 3 + qx)
            global_Y.append(py * 3 + qy)
            global_Z.append(pz * 3 + qz)

    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    # 4. Setup the 3D Plot
    fig = plt.figure(figsize=(12, 10))
    plt.style.use('dark_background')
    ax = fig.add_subplot(111, projection='3d')
    
    def get_color_data(step_idx):
        # Index 2 retrieves the Z-mean
        return history[step_idx, :, :, 2].flatten()

    # Initial scatter plot
    initial_colors = get_color_data(0)
    sc = ax.scatter(global_X, global_Y, global_Z, 
                    c=initial_colors, cmap='coolwarm', 
                    s=60, vmin=-1, vmax=1, alpha=0.8, edgecolors='none')

    # Formatting and Dynamic Axis Scaling
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('<Z> Expectation (Spin State)')
    
    ax.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid) - Step 0", fontsize=14)
    ax.set_xlabel("X (Logical Qubits)")
    ax.set_ylabel("Y (Logical Qubits)")
    ax.set_zlabel("Z (Logical Qubits)")
    
    # Scale axes tightly to the actual deployed volume boundaries
    # Each patch is 3 logical qubits wide
    ax.set_xlim(-0.5, grid_x * 3 - 0.5)
    ax.set_ylim(-0.5, grid_y * 3 - 0.5)
    ax.set_zlim(-0.5, grid_z * 3 - 0.5)

    # Ensure cubic aspect ratio so the lattice doesn't look stretched if grid_z is small
    try:
        ax.set_box_aspect((grid_x, grid_y, max(1, grid_z))) 
    except AttributeError:
        pass # Older matplotlib versions don't support set_box_aspect
        
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

    # 5. Animation Loop
    def update(frame):
        colors = get_color_data(frame)
        sc.set_array(colors)
        ax.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid)\nTrotter Step: {frame}/{num_steps-1}", fontsize=14)
        return sc,

    print("Generating 3D Animation...")
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=False)
    
    plt.show()

if __name__ == "__main__":
    main()
