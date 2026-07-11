import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

# --- CONFIGURATION ---
DATA_FILE = "macroscopic_lattice_states.npy"
CONFIG_FILE = "lattice_config.json"
ENERGY_FILE = "ground_state_energy_curve_multi.csv"
SAVE_FILE = "macroscopic_lattice_anim.mp4"
# ---------------------

def draw_patch_boundaries(ax, grid_x, grid_y, grid_z):
    """Draws faint wireframe cubes representing the boundaries of each 27-qubit patch."""
    for px in range(grid_x):
        for py in range(grid_y):
            for pz in range(grid_z):
                x_start, x_end = px * 3 - 0.5, px * 3 + 2.5
                y_start, y_end = py * 3 - 0.5, py * 3 + 2.5
                z_start, z_end = pz * 3 - 0.5, pz * 3 + 2.5
                
                corners = np.array(list(product([x_start, x_end], [y_start, y_end], [z_start, z_end])))
                
                for s, e in combinations(corners, 2):
                    if np.sum(np.abs(s - e)) == 3.0: 
                        ax.plot3D(*zip(s, e), color='w', alpha=0.1, linewidth=0.5)

def load_energy_data():
    """Extracts the Total_Energy column from the engine's CSV output."""
    energies = []
    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies.append(float(row["Total_Energy"]))
        print(f"Loaded {len(energies)} energy data points from {ENERGY_FILE}.")
    except FileNotFoundError:
        print(f"Warning: {ENERGY_FILE} not found. Energy tracking will be disabled.")
    except Exception as e:
        print(f"Error reading energy file: {e}")
    return energies

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

    # 2. Load the state history & energy data
    print(f"Loading state history from {DATA_FILE}...")
    try:
        history = np.load(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run the hadron engine first.")
        sys.exit(1)

    num_steps = history.shape[0]
    num_patches = history.shape[1]
    
    if num_patches != expected_patches:
        print(f"Warning: Data file has {num_patches} patches, but config expected {expected_patches}.")

    energy_history = load_energy_data()

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
    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        px, py, pz = patch_coords.get(p, (0,0,0))
        for q in range(27):
            qx, qy, qz = q_coords[q]
            global_X.append(px * 3 + qx)
            global_Y.append(py * 3 + qy)
            global_Z.append(pz * 3 + qz)

    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    # 4. Setup the 3D Plot with Expanded Dimensions
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9))
    
    # Push margins outward to maximize the central 3D viewport space
    fig.subplots_adjust(left=0.01, right=0.90, top=0.92, bottom=0.16)
    ax = fig.add_subplot(111, projection='3d')
    
    def get_vector_data(step_idx):
        U = history[step_idx, :, :, 0].flatten()
        V = history[step_idx, :, :, 1].flatten()
        W = history[step_idx, :, :, 2].flatten()
        return U, V, W

    # Render faint anchor points for coordinates
    ax.scatter(global_X, global_Y, global_Z, c='white', s=4, alpha=0.2, edgecolors='none')

    U, V, W = get_vector_data(0)
    
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm(W))

    # Initial Quiver Plot
    quiver_obj = [ax.quiver(global_X, global_Y, global_Z, U, V, W, 
                            length=0.75, colors=colors, arrow_length_ratio=0.3)]

    draw_patch_boundaries(ax, grid_x, grid_y, grid_z)
    
    # Isolate the colorbar to a custom, non-intrusive axis on the far right edge
    ax_cbar = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('<Z> Expectation (Spin State)', fontsize=10)
    
    # Text annotation for HUD
    energy_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, color='lightgreen', fontsize=12, fontweight='bold')
    
    ax.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid) - Step 0", fontsize=14, pad=10)
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

    # 5. Add Mouse Scroll Zoom
    def on_scroll(event):
        if event.inaxes != ax: return
        scale_factor = 1.1 if event.button == 'down' else 1/1.1
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        
        x_c, y_c, z_c = sum(xlim)/2, sum(ylim)/2, sum(zlim)/2
        x_r, y_r, z_r = (xlim[1]-xlim[0])*scale_factor, (ylim[1]-ylim[0])*scale_factor, (zlim[1]-zlim[0])*scale_factor
        
        ax.set_xlim3d([x_c - x_r/2, x_c + x_r/2])
        ax.set_ylim3d([y_c - y_r/2, y_c + y_r/2])
        ax.set_zlim3d([z_c - z_r/2, z_c + z_r/2])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # 6. Low-Profile UI Controls (Lowered and consolidated at the bottom)
    ax_slider = fig.add_axes([0.15, 0.06, 0.70, 0.025])
    slider = Slider(ax=ax_slider, label='Trotter Step', valmin=0, valmax=num_steps-1, valinit=0, valstep=1, color='#4a90e2')

    ax_play = fig.add_axes([0.46, 0.015, 0.08, 0.03])
    btn_play = Button(ax_play, 'Pause', color='#222222', hovercolor='#444444')
    is_playing = [True] 

    # 7. Update Logic
    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)
        
        quiver_obj[0].remove()
        
        colors = cmap(norm(W))
        quiver_obj[0] = ax.quiver(global_X, global_Y, global_Z, U, V, W, 
                                  length=0.75, colors=colors, arrow_length_ratio=0.3)
        
        ax.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid)\nTrotter Step: {frame}/{num_steps-1}", fontsize=14, pad=10)
        if energy_history and frame < len(energy_history):
            energy_text.set_text(f"Total Energy: {energy_history[frame]:.4f}")
        else:
            energy_text.set_text("")
            
        ax.view_init(elev=ax.elev, azim=ax.azim + 0.3)
        
        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True
        
        return quiver_obj[0], energy_text

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

    # 8. Animation & Export
    print("Generating 3D Animation...")
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=False)
    
    print(f"Saving animation to disk as '{SAVE_FILE}' (this may take a minute)...")
    try:
        ani.save(SAVE_FILE, writer='ffmpeg', fps=10)
        print("Save complete.")
    except Exception as e:
        print(f"Failed to save to disk. Error: {e}")

    print("Opening interactive viewer...")
    plt.show()

if __name__ == "__main__":
    main()
