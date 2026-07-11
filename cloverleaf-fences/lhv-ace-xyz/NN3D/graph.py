import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
DATA_FILE = "macroscopic_lattice_states.npy"
CONFIG_FILE = "lattice_config.json"
SAVE_FILE = "macroscopic_lattice_anim.mp4"
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
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    
    # Adjust subplot to make room for slider and button at the bottom
    fig.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(111, projection='3d')
    
    def get_color_data(step_idx):
        # Index 2 retrieves the Z-mean
        return history[step_idx, :, :, 2].flatten()

    initial_colors = get_color_data(0)
    
    # Reverted to fixed absolute scaling [-1.0, 1.0] for true spin representation
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap('coolwarm')

    # Initial scatter plot
    sc = ax.scatter(global_X, global_Y, global_Z, 
                    c=initial_colors, cmap=cmap, norm=norm, 
                    s=60, alpha=0.8, edgecolors='none')

    # Formatting and Dynamic Axis Scaling
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('<Z> Expectation (Spin State)')
    
    ax.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid) - Step 0", fontsize=14)
    ax.set_xlabel("X (Logical Qubits)")
    ax.set_ylabel("Y (Logical Qubits)")
    ax.set_zlabel("Z (Logical Qubits)")
    
    ax.set_xlim(-0.5, grid_x * 3 - 0.5)
    ax.set_ylim(-0.5, grid_y * 3 - 0.5)
    ax.set_zlim(-0.5, grid_z * 3 - 0.5)

    try:
        ax.set_box_aspect((grid_x, grid_y, max(1, grid_z))) 
    except AttributeError:
        pass 
        
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

    # 5. Interactive UI Elements (Slider & Button)
    ax_slider = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider, 
        label='Trotter Step', 
        valmin=0, 
        valmax=num_steps - 1, 
        valinit=0, 
        valstep=1,
        color='#4a90e2'
    )

    ax_play = fig.add_axes([0.85, 0.1, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')
    is_playing = [True] 

    # 6. Update Logic
    def update(frame):
        frame = int(frame)
        colors = get_color_data(frame)
        
        sc.set_array(colors)
        sc._facecolor3d = cmap(norm(colors))
        
        ax.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid)\nTrotter Step: {frame}/{num_steps-1}", fontsize=14)
        
        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True
        
        return sc,

    def on_slider_update(val):
        update(val)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_update)

    def toggle_play(event):
        if is_playing[0]:
            ani.event_source.stop()
            btn_play.label.set_text('Play')
        else:
            ani.event_source.start()
            btn_play.label.set_text('Pause')
        is_playing[0] = not is_playing[0]
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    # 7. Animation & Export
    print("Generating 3D Animation...")
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=False)
    
    print(f"Saving animation to disk as '{SAVE_FILE}' (this may take a minute)...")
    try:
        ani.save(SAVE_FILE, writer='ffmpeg', fps=10)
        print("Save complete.")
    except Exception as e:
        print(f"Failed to save to disk. Make sure 'ffmpeg' is installed. Error: {e}")

    print("Opening interactive viewer...")
    plt.show()

if __name__ == "__main__":
    main()
