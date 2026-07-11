# -*- coding: us-ascii -*-
import sys
import csv
import json
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import product, combinations

# --- CONFIGURATION ---
DATA_FILE = "macroscopic_lattice_states.npy"
CONFIG_FILE = "lattice_config.json"
ENERGY_FILE = "ground_state_energy_curve_multi.csv"
PROFILES_FILE = "boundary_profiles_multi.csv"
SAVE_FILE = "macroscopic_lattice_dash.mp4"
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

def load_analytics_data(num_steps, log_prefix=""):
    """Extracts energy components and boundary profiles for 2D analytics."""
    energies = {'Total': [], 'Bulk': [], 'Boundary': []}
    profiles = {}
    
    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies['Total'].append(float(row["Total_Energy"]))
                energies['Bulk'].append(float(row["Bulk_Energy"]))
                energies['Boundary'].append(float(row["Boundary_Energy"]))
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE} properly: {e}")

    # Ensure energy arrays match the number of steps (pad with NaN if aborted early)
    for key in energies:
        if len(energies[key]) < num_steps:
            energies[key].extend([np.nan] * (num_steps - len(energies[key])))
        energies[key] = energies[key][:num_steps]

    try:
        with open(PROFILES_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s, p, f_name = int(row["Step"]), int(row["Patch"]), row["Face"]
                if s not in profiles: profiles[s] = {}
                if p not in profiles[s]: profiles[s][p] = {}
                profiles[s][p][f_name] = np.array([float(row["X_mean"]), float(row["Y_mean"]), float(row["Z_mean"])])
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {PROFILES_FILE} properly: {e}")

    return energies, profiles

def run_dashboard(mode="interactive"):
    """
    Main visualization routine.
    mode can be "interactive" (opens UI) or "save" (renders to disk headless).
    """
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "
    
    # 1. Dynamically load grid configuration
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            grid_x, grid_y, grid_z = config.get("grid_x", 1), config.get("grid_y", 1), config.get("grid_z", 1)
            expected_patches = config.get("num_patches", grid_x * grid_y * grid_z)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    # 2. Load the state history
    try:
        history = np.load(DATA_FILE)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps, num_patches = history.shape[0], history.shape[1]
    
    # 3. Load Analytics, Calculate Disagreements, and Compute Derivatives
    energies, profiles = load_analytics_data(num_steps, prefix)
    
    patch_coords = {}
    coord_to_patch = {}
    idx = 0
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                patch_coords[idx] = (x, y, z)
                coord_to_patch[(x, y, z)] = idx
                idx += 1

    interfaces = []
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                p1 = coord_to_patch[(x, y, z)]
                if x < grid_x - 1:
                    p2 = coord_to_patch[(x + 1, y, z)]
                    interfaces.append((p1, p2, "+X", "-X", x * 3 + 2.5, y * 3 + 1.0, z * 3 + 1.0))
                if y < grid_y - 1:
                    p2 = coord_to_patch[(x, y + 1, z)]
                    interfaces.append((p1, p2, "+Y", "-Y", x * 3 + 1.0, y * 3 + 2.5, z * 3 + 1.0))
                if z < grid_z - 1:
                    p2 = coord_to_patch[(x, y, z + 1)]
                    interfaces.append((p1, p2, "+Z", "-Z", x * 3 + 1.0, y * 3 + 1.0, z * 3 + 2.5))

    disagreements = np.zeros((num_steps, len(interfaces)))
    for s in range(num_steps):
        if s in profiles:
            for i, (p1, p2, f1, f2, _, _, _) in enumerate(interfaces):
                try:
                    v1, v2 = profiles[s][p1][f1], profiles[s][p2][f2]
                    disagreements[s, i] = np.linalg.norm(v1 - v2)
                except KeyError: pass

    avg_disagreement = np.mean(disagreements, axis=1) if interfaces else np.zeros(num_steps)

    # Compute numerical gradients for the derivative panel
    dE_dt = np.gradient(energies['Total']) if len(energies['Total']) > 1 else np.zeros(num_steps)
    dRes_dt = np.gradient(avg_disagreement) if len(avg_disagreement) > 1 else np.zeros(num_steps)

    # 4. Setup Global 3D Coordinates
    q_coords = {}
    for x in range(3):
        for y in range(3):
            for z in range(3):
                q_coords[x * 9 + y * 3 + z] = (x, y, z)

    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        px, py, pz = patch_coords[p]
        for q in range(27):
            qx, qy, qz = q_coords[q]
            global_X.append(px * 3 + qx)
            global_Y.append(py * 3 + qy)
            global_Z.append(pz * 3 + qz)

    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    # Calculate Macroscopic Patch Centers for Clouds and Network Links
    CX, CY, CZ = [], [], []
    for p in range(num_patches):
        CX.append(patch_coords[p][0] * 3 + 1.0)
        CY.append(patch_coords[p][1] * 3 + 1.0)
        CZ.append(patch_coords[p][2] * 3 + 1.0)

    # 5. Initialize UI and GridSpec Layout
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)
    
    ax3d = fig.add_subplot(gs[0], projection='3d')
    # Expanded GridSpec to 3 rows to accommodate the new Derivatives panel
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.4)
    ax_energy = fig.add_subplot(gs_right[0])
    ax_dis = fig.add_subplot(gs_right[1])
    ax_deriv = fig.add_subplot(gs_right[2])

    def get_vector_data(step_idx):
        return history[step_idx, :, :, 0].flatten(), history[step_idx, :, :, 1].flatten(), history[step_idx, :, :, 2].flatten()

    ax3d.scatter(global_X, global_Y, global_Z, c='white', s=4, alpha=0.15, edgecolors='none')
    U, V, W = get_vector_data(0)
    
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap('coolwarm')
    quiver_obj = [ax3d.quiver(global_X, global_Y, global_Z, U, V, W, length=0.75, colors=cmap(norm(W)), arrow_length_ratio=0.3)]
    draw_patch_boundaries(ax3d, grid_x, grid_y, grid_z)
    
    mean_Z_per_patch = np.mean(history[:, :, :, 2], axis=2)
    cloud = ax3d.scatter(CX, CY, CZ, s=2500, c=mean_Z_per_patch[0], cmap=cmap, norm=norm, alpha=0.15, edgecolors='none')

    # Correlation Network (Replaces isolated markers with a dynamic 3D web)
    network_lines = []
    if interfaces:
        for i, (p1, p2, _, _, _, _, _) in enumerate(interfaces):
            network_lines.append([(CX[p1], CY[p1], CZ[p1]), (CX[p2], CY[p2], CZ[p2])])
        
        dis_cmap = plt.get_cmap('RdYlGn_r')
        dis_norm = mcolors.Normalize(vmin=0.0, vmax=2.0)
        
        line_collection = Line3DCollection(network_lines, cmap=dis_cmap, norm=dis_norm, linewidths=2.5, alpha=0.8)
        line_collection.set_array(disagreements[0])
        ax3d.add_collection3d(line_collection)

    # 6. Add Colorbar Legend on the Far Left
    ax_cbar = fig.add_axes([0.02, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('<Z> Expectation (Spin State)', fontsize=10)

    energy_text = ax3d.text2D(0.04, 0.96, "", transform=ax3d.transAxes, color='lightgreen', fontsize=12, fontweight='bold')
    
    ax3d.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid)\nTrotter Step: 0/{num_steps-1}", fontsize=14, pad=10)
    ax3d.set_xlim(-0.5, grid_x * 3 - 0.5)
    ax3d.set_ylim(-0.5, grid_y * 3 - 0.5)
    ax3d.set_zlim(-0.5, grid_z * 3 - 0.5)
    try: ax3d.set_box_aspect((grid_x, grid_y, max(1, grid_z))) 
    except AttributeError: pass 
    
    ax3d.xaxis.pane.fill, ax3d.yaxis.pane.fill, ax3d.zaxis.pane.fill = False, False, False
    ax3d.grid(color='grey', linestyle='--', linewidth=0.3, alpha=0.5)

    # --- 2D Analytics Assembly ---
    ax_energy.plot(energies['Total'], label='Total Energy', color='lightgreen')
    ax_energy.plot(energies['Bulk'], label='Bulk', color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Boundary', color='orange')
    ax_energy.set_title("Energy Components")
    ax_energy.legend(fontsize=8, loc='upper left')
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    if interfaces:
        ax_dis.plot(avg_disagreement, color='crimson', label='Mean Interface Disagreement')
        ax_dis.set_title("Boundary Polarization Residuals")
        ax_dis.legend(fontsize=8, loc='upper left')
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # New Derivatives Panel (Dual-axis)
    ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("Energy Delta")
    ax_deriv.legend(loc='upper left', fontsize=8)
    
    ax_deriv_r = ax_deriv.twinx()
    ax_deriv_r.plot(dRes_dt, label='dResidual/dt', color='crimson')
    ax_deriv_r.set_ylabel("Residual Delta")
    ax_deriv_r.legend(loc='upper right', fontsize=8)
    
    ax_deriv.set_title("Derivatives (Convergence Rate)")
    ax_deriv.grid(True, alpha=0.2)
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # Adjust layout to make room for the new left-aligned colorbar
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    ax_slider = fig.add_axes([0.15, 0.05, 0.60, 0.02])
    slider = Slider(ax=ax_slider, label='Trotter Step', valmin=0, valmax=num_steps-1, valinit=0, valstep=1, color='#4a90e2')

    ax_play = fig.add_axes([0.80, 0.035, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')
    is_playing = [True] 

    def on_scroll(event):
        if event.inaxes != ax3d: return
        scale_factor = 1.1 if event.button == 'down' else 1/1.1
        xlim, ylim, zlim = ax3d.get_xlim3d(), ax3d.get_ylim3d(), ax3d.get_zlim3d()
        x_c, y_c, z_c = sum(xlim)/2, sum(ylim)/2, sum(zlim)/2
        x_r, y_r, z_r = (xlim[1]-xlim[0])*scale_factor, (ylim[1]-ylim[0])*scale_factor, (zlim[1]-zlim[0])*scale_factor
        ax3d.set_xlim3d([x_c - x_r/2, x_c + x_r/2])
        ax3d.set_ylim3d([y_c - y_r/2, y_c + y_r/2])
        ax3d.set_zlim3d([z_c - z_r/2, z_c + z_r/2])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)
        
        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(global_X, global_Y, global_Z, U, V, W, length=0.75, colors=cmap(norm(W)), arrow_length_ratio=0.3)
        cloud.set_array(mean_Z_per_patch[frame])
        
        if interfaces:
            line_collection.set_array(disagreements[frame])
            
        ax3d.set_title(f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid)\nTrotter Step: {frame}/{num_steps-1}", fontsize=14, pad=10)
        
        if len(energies['Total']) > 0 and frame < len(energies['Total']):
            e_val = energies['Total'][frame]
            if not np.isnan(e_val):
                energy_text.set_text(f"Total Energy: {e_val:.4f}")
            else:
                energy_text.set_text("")
        else:
            energy_text.set_text("")
            
        if len(energies['Total']) > 0: vline_e.set_xdata([frame, frame])
        if interfaces: 
            vline_d.set_xdata([frame, frame])
            vline_deriv.set_xdata([frame, frame])
            
        ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)
        
        # Visually update slider during MP4 save rendering
        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True
        
        return quiver_obj[0], cloud, energy_text, line_collection

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

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=False)
    
    if mode == "save":
        print(f"{prefix}Commencing 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            # DPI 216 on 18x10 figsize forces 3888x2160 UHD output
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Save complete.")
        except Exception as e:
            print(f"{prefix}Failed to save to disk. Make sure 'ffmpeg' is installed. Error: {e}")
    else:
        print(f"{prefix}Opening GUI...")
        plt.show()

def main():
    # Enforce spawn mode to guarantee clean Matplotlib context boundaries across processes
    mp.set_start_method('spawn', force=True)
    
    print("Forking 4K render to background process...")
    render_process = mp.Process(target=run_dashboard, args=("save",))
    render_process.start()

    # Launch interactive viewer immediately on the main thread
    run_dashboard(mode="interactive")

    # If the user closes the Matplotlib window before the background render is finished
    if render_process.is_alive():
        print("\nInteractive viewer closed. Waiting for the background 4K render to finish...")
        render_process.join()
        
    print("All processes terminated.")

if __name__ == "__main__":
    main()
