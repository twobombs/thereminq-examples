# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_v5.py
# Changes vs v4:
#   - Fourth heatmap panel added: ⟨X⟩+⟨Y⟩+⟨Z⟩ total polarization sum,
#     normalised over [-3, +3] with its own distinct colormap (viridis)
#     so it reads visually differently from the per-component panels.
#   - GridSpec right column expanded to 7 rows to fit the new panel.
#   - _init_heatmap() accepts an explicit norm so each panel can use its own.
#   - update() computes and pushes the sum slice each frame.
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

# --- CONFIGURATION ---
DATA_FILE = "macroscopic_lattice_states.npy"
CONFIG_FILE = "lattice_config.json"
ENERGY_FILE = "ground_state_energy_curve_multi.csv"
PROFILES_FILE = "boundary_profiles_multi.csv"
SAVE_FILE = "macroscopic_lattice_dash.mp4"
# ---------------------



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
                s = int(row["Step"])
                p = int(row["Patch"])
                f_name = row["Face"]
                if s not in profiles:
                    profiles[s] = {}
                if p not in profiles[s]:
                    profiles[s][p] = {}
                profiles[s][p][f_name] = np.array([
                    float(row["X_mean"]), float(row["Y_mean"]), float(row["Z_mean"])
                ])
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {PROFILES_FILE} properly: {e}")

    return energies, profiles


def _safe_gradient(arr_with_nans):
    """
    Compute np.gradient only over the leading valid (non-NaN) slice.
    Returns a full-length array with NaN in positions where input was NaN.
    Prevents gradient NaN bleed at the boundary caused by trailing NaN padding.
    """
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_count = int(np.sum(valid_mask))
    if valid_count > 1:
        # Assume valid values are a contiguous leading block (NaN padding at the tail)
        out[:valid_count] = np.gradient(arr[:valid_count])
    elif valid_count == 1:
        out[:1] = 0.0
    return out


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
            grid_x = config.get("grid_x", 1)
            grid_y = config.get("grid_y", 1)
            grid_z = config.get("grid_z", 1)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    # 2. Load the state history.
    # FIX 3: background render uses mmap_mode='r' to avoid loading the full
    # array into RAM twice when the interactive viewer is also running.
    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps, num_patches = history.shape[0], history.shape[1]
    total_qubits = num_patches * 27

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

    disagreements = np.zeros((num_steps, max(len(interfaces), 1)))
    for s in range(num_steps):
        if s in profiles:
            for i, (p1, p2, f1, f2, _, _, _) in enumerate(interfaces):
                try:
                    v1 = profiles[s][p1][f1]
                    v2 = profiles[s][p2][f2]
                    disagreements[s, i] = np.linalg.norm(v1 - v2)
                except KeyError:
                    pass

    avg_disagreement = np.mean(disagreements, axis=1) if interfaces else np.zeros(num_steps)

    # FIX 4: use _safe_gradient so NaN-padded tails don't bleed into valid values
    dE_dt = _safe_gradient(energies['Total'])
    dRes_dt = _safe_gradient(avg_disagreement)

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

    CX = [patch_coords[p][0] * 3 + 1.0 for p in range(num_patches)]
    CY = [patch_coords[p][1] * 3 + 1.0 for p in range(num_patches)]
    CZ = [patch_coords[p][2] * 3 + 1.0 for p in range(num_patches)]

    # 5. Initialise UI and GridSpec Layout
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)

    ax3d = fig.add_subplot(gs[0], projection='3d')
    # 7 rows: energy, disagreement, derivatives, heatmap-X, heatmap-Y, heatmap-Z, heatmap-Sum
    gs_right = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=gs[1], hspace=0.80)
    ax_energy   = fig.add_subplot(gs_right[0])
    ax_dis      = fig.add_subplot(gs_right[1])
    ax_deriv    = fig.add_subplot(gs_right[2])
    ax_hmap_x   = fig.add_subplot(gs_right[3])
    ax_hmap_y   = fig.add_subplot(gs_right[4])
    ax_hmap_z   = fig.add_subplot(gs_right[5])
    ax_hmap_sum = fig.add_subplot(gs_right[6])

    def get_vector_data(step_idx):
        return (
            history[step_idx, :, :, 0].flatten(),
            history[step_idx, :, :, 1].flatten(),
            history[step_idx, :, :, 2].flatten(),
        )

    U, V, W = get_vector_data(0)
    # Normaliser shared by quiver colours, colourbar, and all three heatmaps
    spin_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    vector_colors = [
        (0.15, 0.35, 0.85, 0.85),  # spin-down  (-1) → blue
        (0.85, 0.85, 0.85, 0.45),  # equatorial ( 0) → grey
        (0.85, 0.15, 0.25, 0.85),  # spin-up    (+1) → red
    ]
    vector_cmap = mcolors.LinearSegmentedColormap.from_list("ghost_vectors", vector_colors)

    def _quiver_colors(w_flat):
        """Map the Z-component array to per-arrow RGBA using the shared norm+cmap."""
        return vector_cmap(spin_norm(w_flat))

    quiver_obj = [ax3d.quiver(
        global_X, global_Y, global_Z, U, V, W,
        length=0.75, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    # Patch boundary wireframes, cloud scatter, and correlation network are
    # intentionally omitted — the view shows arrows only.

    # 6. Colorbar — driven by spin_norm so ticks span the true [-1, +1] range
    ax_cbar = fig.add_axes([0.02, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('<Z> Expectation (Spin State)', fontsize=10)

    energy_text = ax3d.text2D(
        0.04, 0.96, "", transform=ax3d.transAxes,
        color='lightgreen', fontsize=12, fontweight='bold'
    )

    ax3d.set_title(
        f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
        f"{num_patches} Patches | {total_qubits} Qubits)\nTrotter Step: 0/{num_steps-1}",
        fontsize=14, pad=10
    )
    ax3d.set_xlim(-0.5, grid_x * 3 - 0.5)
    ax3d.set_ylim(-0.5, grid_y * 3 - 0.5)
    ax3d.set_zlim(-0.5, grid_z * 3 - 0.5)
    try:
        ax3d.set_box_aspect((grid_x, grid_y, max(1, grid_z)))
    except AttributeError:
        pass  # set_box_aspect requires matplotlib >= 3.3

    # Hide all pane surfaces, pane edges, tick lines, tick labels, and the 3D grid
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.line.set_linewidth(0)
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()

    # --- 2D Analytics ---
    ax_energy.plot(energies['Total'], label='Total Energy', color='lightgreen')
    ax_energy.plot(energies['Bulk'], label='Bulk', color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Boundary', color='orange')
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.legend(fontsize=8, loc='upper left')
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # FIX 2: always initialise vline_d / vline_deriv so update() never hits NameError
    vline_d = None
    vline_deriv = None

    if interfaces:
        ax_dis.plot(avg_disagreement, color='crimson', label='Mean Interface Disagreement')
        ax_dis.set_title("Boundary Polarization Residuals", fontsize=10)
        ax_dis.legend(fontsize=8, loc='upper left')
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("Energy Delta", fontsize=8)
    ax_deriv.legend(loc='upper left', fontsize=8)

    ax_deriv_r = ax_deriv.twinx()
    ax_deriv_r.plot(dRes_dt, label='dResidual/dt', color='crimson')
    ax_deriv_r.set_ylabel("Residual Delta", fontsize=8)
    ax_deriv_r.legend(loc='upper right', fontsize=8)

    ax_deriv.set_title("Derivatives (Convergence Rate)", fontsize=10)
    ax_deriv.grid(True, alpha=0.2)
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # 7. Three stacked heatmaps — one per Bloch-sphere component (X, Y, Z)
    heatmap_colors = [
        (0.15, 0.35, 0.85, 1.0),   # -1 → blue
        (0.10, 0.10, 0.10, 1.0),   #  0 → near-black
        (0.85, 0.15, 0.25, 1.0),   # +1 → red
    ]
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list("heatmap_cmap", heatmap_colors)

    # Build shared y-tick labels (only on the top panel to save space)
    y_ticks = []
    y_labels = []
    if num_patches <= 32:
        y_ticks = list(np.arange(num_patches))
        y_labels = [
            f"X:{patch_coords[i][0]} Y:{patch_coords[i][1]} Z:{patch_coords[i][2]}"
            for i in y_ticks
        ]
    else:
        z_mid = grid_z // 2
        for i in range(num_patches):
            if patch_coords[i][2] == z_mid:
                y_ticks.append(i)
                y_labels.append(
                    f"X:{patch_coords[i][0]} Y:{patch_coords[i][1]} Z:{patch_coords[i][2]}"
                )

    def _init_heatmap(ax, data, cmap, norm, label, show_ylabel=False, show_xlabel=False):
        """Create an imshow panel for the given data slice."""
        img = ax.imshow(
            data, cmap=cmap, norm=norm,
            aspect='auto', interpolation='nearest'
        )
        for i in range(1, num_patches):
            if patch_coords[i][0] != patch_coords[i - 1][0]:
                ax.axhline(i - 0.5, color='white', linewidth=1.2, alpha=1.0)
            elif patch_coords[i][1] != patch_coords[i - 1][1]:
                ax.axhline(i - 0.5, color='#aaaaaa', linewidth=0.7, alpha=0.7, linestyle=':')
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=6)
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel("Local Qubit Index (0-26)", fontsize=8)
            ax.tick_params(axis='x', which='major', labelsize=7)
        else:
            ax.set_xticks([])
        return img

    # Per-component norm: each axis spans [-1, +1]
    comp_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0],
                            heatmap_cmap, comp_norm, "Polarization \u27e8X\u27e9",
                            show_ylabel=True,  show_xlabel=False)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1],
                            heatmap_cmap, comp_norm, "Polarization \u27e8Y\u27e9",
                            show_ylabel=False, show_xlabel=False)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2],
                            heatmap_cmap, comp_norm, "Polarization \u27e8Z\u27e9",
                            show_ylabel=False, show_xlabel=False)

    # Sum panel: ⟨X⟩+⟨Y⟩+⟨Z⟩, range [-3, +3]; viridis distinguishes it from components
    sum_norm = mcolors.Normalize(vmin=-3.0, vmax=3.0)
    sum_data_0 = history[0, :, :, 0] + history[0, :, :, 1] + history[0, :, :, 2]
    hmap_sum = _init_heatmap(ax_hmap_sum, sum_data_0,
                              plt.get_cmap('viridis'), sum_norm,
                              "Total Polarization \u27e8X\u27e9+\u27e8Y\u27e9+\u27e8Z\u27e9  [-3…+3]",
                              show_ylabel=False, show_xlabel=True)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    ax_slider = fig.add_axes([0.15, 0.05, 0.60, 0.02])
    slider = Slider(
        ax=ax_slider, label='Trotter Step',
        valmin=0, valmax=num_steps - 1, valinit=0, valstep=1, color='#4a90e2'
    )

    ax_play = fig.add_axes([0.80, 0.035, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')

    # FIX 7: use a plain bool with nonlocal instead of the mutable-list closure hack
    is_playing = True

    # FIX 5: true 3D perspective zoom via ax3d.dist instead of axis-limit manipulation
    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        ax3d.dist *= 1.1 if event.button == 'down' else 0.9
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key_press(event):
        if event.key == ' ':
            current_step = int(slider.val)
            e_list = energies['Total']
            e_val = e_list[current_step] if current_step < len(e_list) else float('nan')
            e_str = f"{e_val:.4f}" if (isinstance(e_val, float) and not np.isnan(e_val)) else "NaN"
            filename = (
                f"dash_snapshot_{grid_x}x{grid_y}x{grid_z}_step{current_step}_E{e_str}.png"
            )
            print(f"{prefix}Saving screenshot to {filename}...")
            fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"{prefix}Screenshot saved.")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # FIX 6: track whether the current call is from FuncAnimation or from the slider
    _from_animation = [False]

    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)

        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            global_X, global_Y, global_Z, U, V, W,
            length=0.75, colors=_quiver_colors(W), arrow_length_ratio=0.3
        )

        ax3d.set_title(
            f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
            f"{num_patches} Patches | {total_qubits} Qubits)\nTrotter Step: {frame}/{num_steps-1}",
            fontsize=14, pad=10
        )

        e_list = energies['Total']
        if e_list and frame < len(e_list):
            e_val = e_list[frame]
            energy_text.set_text(f"Total Energy: {e_val:.4f}" if not np.isnan(e_val) else "")
        else:
            energy_text.set_text("")

        vline_e.set_xdata([frame, frame])
        if vline_d is not None:
            vline_d.set_xdata([frame, frame])
        if vline_deriv is not None:
            vline_deriv.set_xdata([frame, frame])

        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])
        hmap_sum.set_data(
            history[frame, :, :, 0] + history[frame, :, :, 1] + history[frame, :, :, 2]
        )

        # FIX 6: only auto-rotate during playback, not when the user scrubs the slider
        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return quiver_obj[0], energy_text, hmap_x, hmap_y, hmap_z, hmap_sum

    def _animation_update(frame):
        _from_animation[0] = True
        result = update(frame)
        _from_animation[0] = False
        return result

    def on_slider_update(val):
        _from_animation[0] = False
        update(val)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_update)

    def toggle_play(event):
        nonlocal is_playing  # FIX 7
        if is_playing:
            ani.event_source.stop()
            btn_play.label.set_text('Play')
        else:
            ani.event_source.start()
            btn_play.label.set_text('Pause')
        is_playing = not is_playing
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    # blit=False is correct: Axes3D does not support blitting.
    ani = animation.FuncAnimation(
        fig, _animation_update, frames=num_steps, interval=150, blit=False
    )

    if mode == "save":
        print(f"{prefix}Commencing 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Save complete.")
        except Exception as e:
            print(f"{prefix}Failed to save. Is ffmpeg installed? Error: {e}")
    else:
        print(f"{prefix}Opening GUI...")
        plt.show()


def main():
    mp.set_start_method('spawn', force=True)

    print("Forking 4K render to background process...")
    render_process = mp.Process(target=run_dashboard, args=("save",))
    render_process.start()

    run_dashboard(mode="interactive")

    if render_process.is_alive():
        print("\nInteractive viewer closed. Waiting for the background 4K render to finish...")
        render_process.join()

    print("All processes terminated.")


if __name__ == "__main__":
    main()
