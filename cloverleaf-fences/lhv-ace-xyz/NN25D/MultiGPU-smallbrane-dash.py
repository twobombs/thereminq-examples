# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_v7_brane.py
# Dashboard adapted for the Rev 88-B BRANE STACK engine variant
# (16-qubit 4x4 flat tiles, 4 branes stacked along Z, 64 qubits total).
#
# CHANGES vs v6:
#   - GEOMETRY: replaces the hardcoded 3x3x3 27-qubit cube layout with the
#     4x4 planar brane tile (idx = x * 4 + y, matching
#     generate_16q_brane_tile()). Branes are drawn as flat 4x4 sheets
#     separated by LAYER_SPACING along Z so the inter-brane gap is visible.
#   - CONFIG DRIVEN: qubits_per_patch, tile_geometry, and periodic_z are
#     read from lattice_config.json (with fallbacks). total_qubits and the
#     heatmap x-axis label are derived, not hardcoded to 27.
#   - INTERFACES: rebuilt from Z-adjacency of the stack: (z, z+1) for each
#     gap, plus the (grid_z-1, 0) wrap when periodic_z is true and
#     grid_z > 2 (mirroring the engine's GRID_Z=2 double-count guard).
#   - SITE-RESOLVED COUPLING ERROR: the Rev 88-B engine couples qubit (x,y)
#     in layer k to qubit (x,y) in layers k+/-1, and its profiles CSV only
#     carries ONE tile-averaged "BRANE" row per patch. A face-averaged
#     disagreement would erase exactly the in-plane structure the engine
#     was changed to preserve, so the coupling error is now computed
#     directly from macroscopic_lattice_states.npy:
#         D(t, iface) = mean_i ||s_i^(p1)(t) - s_i^(p2)(t)||_2
#     over the 16 aligned sites i, with the full XYZ vector per site.
#     boundary_profiles_multi.csv is therefore no longer needed here.
#   - INTERFACE PLANES: faint static translucent sheets are drawn at the
#     Z-midpoint of each open interface as a visual cue for the coupled
#     gaps (the periodic wrap interface is counted in the error metric but
#     not drawn).
#   - HEATMAPS: 16 columns (Local Qubit Index 0-15); one row per brane,
#     labelled "Layer z"; a thin separator line between every row since
#     each row is a distinct brane.
#
# Unchanged vs v6:
#   - ENERGY_FILE name and MeanField_* column names (Rev 88-B writes the
#     same fieldnames as Rev 85+).
#   - _safe_gradient NaN-padding handling, slider/play/scroll/snapshot UI,
#     dual interactive + background-4K-render process model, blit=False.

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- CONFIGURATION ---
DATA_FILE     = "macroscopic_lattice_states.npy"
CONFIG_FILE   = "lattice_config.json"
ENERGY_FILE   = "meanfield_ground_state_energy_curve_multi.csv"
SAVE_FILE     = "macroscopic_lattice_dash_brane.mp4"

TILE_LX, TILE_LY = 4, 4        # in-plane brane tile extent (matches engine)
LAYER_SPACING    = 2.0         # visual Z gap between stacked branes
# ---------------------


def load_energy_data(num_steps, log_prefix=""):
    """Extracts energy components for the 2D analytics panels."""
    energies = {'Total': [], 'Bulk': [], 'Boundary': []}

    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies['Total'].append(float(row["MeanField_Total_Energy"]))
                energies['Bulk'].append(float(row["MeanField_Bulk_Energy"]))
                energies['Boundary'].append(float(row["MeanField_Boundary_Energy"]))
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE} properly: {e}")

    # Ensure energy arrays match the number of steps (pad with NaN if aborted early)
    for key in energies:
        if len(energies[key]) < num_steps:
            energies[key].extend([np.nan] * (num_steps - len(energies[key])))
        energies[key] = energies[key][:num_steps]

    return energies


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

    # 1. Dynamically load stack configuration
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    grid_z         = config.get("grid_z", 4)
    periodic_z     = bool(config.get("periodic_z", False))
    cfg_qpp        = config.get("qubits_per_patch", TILE_LX * TILE_LY)
    tile_geometry  = config.get("tile_geometry", "4x4_brane_stack")

    if tile_geometry != "4x4_brane_stack":
        print(f"{prefix}Warning: tile_geometry is '{tile_geometry}', "
              f"expected '4x4_brane_stack'. Layout may be wrong.")

    # 2. Load the state history: shape (n_steps, patch, qubit, XYZ)
    # background render uses mmap_mode='r' to avoid loading the full
    # array into RAM twice when the interactive viewer is also running.
    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps, num_patches, qpp = history.shape[0], history.shape[1], history.shape[2]
    if qpp != cfg_qpp:
        print(f"{prefix}Warning: config qubits_per_patch={cfg_qpp} but state "
              f"dump has {qpp}; trusting the state dump.")
    if num_patches != grid_z:
        print(f"{prefix}Warning: config grid_z={grid_z} but state dump has "
              f"{num_patches} patches; trusting the state dump.")
        grid_z = num_patches
    total_qubits = num_patches * qpp

    # 3. Load energies, build interfaces, compute site-resolved coupling error
    energies = load_energy_data(num_steps, prefix)

    # Brane-stack layout: patch index == layer z (matches engine patch_coords)
    interfaces = [(z, z + 1) for z in range(grid_z - 1)]
    wrap_interface = None
    if periodic_z and grid_z > 2:
        # Mirrors the engine's GRID_Z=2 double-count guard
        wrap_interface = (grid_z - 1, 0)
        interfaces.append(wrap_interface)

    # Site-resolved inter-brane coupling error, straight from the state dump:
    #   D[t, iface] = mean over the 16 aligned sites of the Euclidean norm
    #                 of the per-site Bloch-vector difference.
    # NOTE: this is NOT a residual against any external reference. It measures
    # internal self-consistency of the site-resolved inter-brane coupling
    # (how well the per-site kick field is stitching independently-evolved
    # branes together), NOT simulation accuracy relative to a ground truth.
    n_ifaces = max(len(interfaces), 1)
    disagreements = np.zeros((num_steps, n_ifaces))
    if interfaces:
        hist_arr = np.asarray(history)  # mmap-safe view for vectorised math
        for i, (p1, p2) in enumerate(interfaces):
            diff = hist_arr[:, p1, :, :] - hist_arr[:, p2, :, :]   # (steps, 16, 3)
            disagreements[:, i] = np.mean(np.linalg.norm(diff, axis=2), axis=1)

    avg_disagreement = np.mean(disagreements, axis=1) if interfaces else np.zeros(num_steps)

    # _safe_gradient: avoids NaN bleed from trailing padding into valid values.
    dE_dt   = _safe_gradient(energies['Total'])
    dRes_dt = _safe_gradient(avg_disagreement)

    # 4. Setup Global 3D Coordinates
    # Brane tile: idx = x * TILE_LY + y (row-major in x), z = 0 within tile.
    # Layer p sits at Z = p * LAYER_SPACING.
    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        for q in range(qpp):
            qx, qy = divmod(q, TILE_LY)
            global_X.append(float(qx))
            global_Y.append(float(qy))
            global_Z.append(p * LAYER_SPACING)

    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    # 5. Initialise UI and GridSpec Layout
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)

    ax3d = fig.add_subplot(gs[0], projection='3d')
    # 6 rows: energy, disagreement, derivatives, heatmap-X, heatmap-Y, heatmap-Z
    gs_right = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[1], hspace=0.80)
    ax_energy   = fig.add_subplot(gs_right[0])
    ax_dis      = fig.add_subplot(gs_right[1])
    ax_deriv    = fig.add_subplot(gs_right[2])
    ax_hmap_x   = fig.add_subplot(gs_right[3])
    ax_hmap_y   = fig.add_subplot(gs_right[4])
    ax_hmap_z   = fig.add_subplot(gs_right[5])

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
        (0.15, 0.35, 0.85, 0.85),  # spin-down  (-1) = blue
        (0.85, 0.85, 0.85, 0.45),  # equatorial ( 0) = grey
        (0.85, 0.15, 0.25, 0.85),  # spin-up    (+1) = red
    ]
    vector_cmap = mcolors.LinearSegmentedColormap.from_list("ghost_vectors", vector_colors)

    def _quiver_colors(w_flat):
        """Map the Z-component array to per-arrow RGBA using the shared norm+cmap."""
        return vector_cmap(spin_norm(w_flat))

    quiver_obj = [ax3d.quiver(
        global_X, global_Y, global_Z, U, V, W,
        length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    # Faint translucent sheets at the midpoint of each open interface gap,
    # as a visual cue for the coupled brane pairs. The periodic wrap
    # interface (if any) is included in the error metric but not drawn.
    margin = 0.4
    for (p1, p2) in interfaces:
        if wrap_interface is not None and (p1, p2) == wrap_interface:
            continue
        z_mid = (p1 + p2) / 2.0 * LAYER_SPACING
        verts = [[
            (-margin,              -margin,              z_mid),
            (TILE_LX - 1 + margin, -margin,              z_mid),
            (TILE_LX - 1 + margin, TILE_LY - 1 + margin, z_mid),
            (-margin,               TILE_LY - 1 + margin, z_mid),
        ]]
        plane = Poly3DCollection(verts, alpha=0.06, facecolor='#4a90e2',
                                 edgecolor='#4a90e2', linewidths=0.5)
        ax3d.add_collection3d(plane)

    # 6. Colorbar - driven by spin_norm so ticks span the true [-1, +1] range
    ax_cbar = fig.add_axes([0.02, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('<Z> Expectation (Spin State)', fontsize=10)

    energy_text = ax3d.text2D(
        0.04, 0.96, "", transform=ax3d.transAxes,
        color='lightgreen', fontsize=12, fontweight='bold'
    )

    stack_desc = f"1x1x{grid_z}{' periodic' if periodic_z else ''}"
    ax3d.set_title(
        f"Brane-Stack Annealing ({stack_desc} | {num_patches} Branes | "
        f"{total_qubits} Qubits)\nTrotter Step: 0/{num_steps-1}",
        fontsize=14, pad=10
    )
    ax3d.set_xlim(-0.5, TILE_LX - 0.5)
    ax3d.set_ylim(-0.5, TILE_LY - 0.5)
    ax3d.set_zlim(-0.5, (grid_z - 1) * LAYER_SPACING + 0.5)
    try:
        ax3d.set_box_aspect((TILE_LX, TILE_LY,
                             max(1.0, (grid_z - 1) * LAYER_SPACING + 1.0)))
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
    ax_energy.plot(energies['Boundary'], label='Inter-Brane', color='orange')
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.legend(fontsize=8, loc='upper left')
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # always initialise vline_d / vline_deriv so update() never hits NameError
    vline_d = None
    vline_deriv = None

    if not interfaces:
        ax_dis.set_visible(False)
    else:
        ax_dis.plot(avg_disagreement, color='crimson',
                    label='Mean site-resolved ||ds||')
        ax_dis.set_title("Inter-Brane Coupling Error (site-resolved)\n"
                         "mean_ifaces( mean_i ||s_i^(k) - s_i^(k+1)|| )",
                         fontsize=9)
        ax_dis.legend(fontsize=8, loc='upper left')
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("Energy Delta", fontsize=8)
    ax_deriv.legend(loc='upper left', fontsize=8)

    ax_deriv_r = ax_deriv.twinx()
    ax_deriv_r.plot(dRes_dt, label='d||ds||/dt', color='crimson')
    ax_deriv_r.set_ylabel("Coupling Error dt", fontsize=8)
    ax_deriv_r.legend(loc='upper right', fontsize=8)

    ax_deriv.set_title("Derivatives (Convergence Rate)", fontsize=10)
    ax_deriv.grid(True, alpha=0.2)
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # 7. Three stacked heatmaps - one per Bloch-sphere component (X, Y, Z)
    heatmap_colors = [
        (0.15, 0.35, 0.85, 1.0),   # -1 = blue
        (0.10, 0.10, 0.10, 1.0),   #  0 = near-black
        (0.85, 0.15, 0.25, 1.0),   # +1 = red
    ]
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list("heatmap_cmap", heatmap_colors)

    # One heatmap row per brane
    y_ticks  = list(np.arange(num_patches))
    y_labels = [f"Layer {i}" for i in y_ticks]

    def _init_heatmap(ax, data, cmap, norm, label, show_ylabel=False, show_xlabel=False):
        """Create an imshow panel for the given data slice."""
        img = ax.imshow(
            data, cmap=cmap, norm=norm,
            aspect='auto', interpolation='nearest'
        )
        # Every row is a distinct brane: separator between all rows
        for i in range(1, num_patches):
            ax.axhline(i - 0.5, color='white', linewidth=0.8, alpha=0.8)
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=6)
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel(f"Local Qubit Index (0-{qpp - 1})", fontsize=8)
            ax.tick_params(axis='x', which='major', labelsize=7)
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0],
                            heatmap_cmap, spin_norm, "Polarization <X>",
                            show_ylabel=True, show_xlabel=False)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1],
                            heatmap_cmap, spin_norm, "Polarization <Y>",
                            show_ylabel=True, show_xlabel=False)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2],
                            heatmap_cmap, spin_norm, "Polarization <Z>",
                            show_ylabel=True, show_xlabel=True)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    ax_slider = fig.add_axes([0.15, 0.05, 0.60, 0.02])
    slider = Slider(
        ax=ax_slider, label='Trotter Step',
        valmin=0, valmax=num_steps - 1, valinit=0, valstep=1, color='#4a90e2'
    )

    ax_play = fig.add_axes([0.80, 0.035, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')

    is_playing = True

    def on_scroll(event):
        if event.inaxes != ax3d:
            return

        # Matplotlib 3.8+ compatible zoom: scroll down zooms out (box shrinks)
        if not hasattr(ax3d, 'custom_zoom'):
            ax3d.custom_zoom = 1.0
        ax3d.custom_zoom *= 0.9 if event.button == 'down' else 1.1

        try:
            current_aspect = ax3d.get_box_aspect()
            if current_aspect is None:
                current_aspect = (TILE_LX, TILE_LY,
                                  max(1.0, (grid_z - 1) * LAYER_SPACING + 1.0))
            ax3d.set_box_aspect(current_aspect, zoom=ax3d.custom_zoom)
        except TypeError:
            # Fallback for Matplotlib < 3.8 where zoom kwarg doesn't exist
            # Scroll down zooms out (distance increases)
            ax3d.dist *= 0.9 if event.button == 'down' else 1.1

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key_press(event):
        if event.key == ' ':
            current_step = int(slider.val)
            e_list = energies['Total']
            e_val = e_list[current_step] if current_step < len(e_list) else float('nan')
            e_str = f"{e_val:.4f}" if (isinstance(e_val, float) and not np.isnan(e_val)) else "NaN"
            filename = (
                f"dash_snapshot_brane_1x1x{grid_z}_step{current_step}_E{e_str}.png"
            )
            print(f"{prefix}Saving screenshot to {filename}...")
            fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"{prefix}Screenshot saved.")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    _from_animation = [False]

    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)

        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            global_X, global_Y, global_Z, U, V, W,
            length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3
        )

        ax3d.set_title(
            f"Brane-Stack Annealing ({stack_desc} | {num_patches} Branes | "
            f"{total_qubits} Qubits)\nTrotter Step: {frame}/{num_steps-1}",
            fontsize=14, pad=10
        )

        e_list = energies['Total']
        if e_list and frame < len(e_list):
            e_val = e_list[frame]
            energy_text.set_text(f"Total Energy: {e_val:.4f}" if not np.isnan(e_val) else "")
        else:
            energy_text.set_text("")

        vline_e.set_xdata([frame])
        if vline_d is not None:
            vline_d.set_xdata([frame])
        if vline_deriv is not None:
            vline_deriv.set_xdata([frame])

        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])

        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return quiver_obj[0], energy_text, hmap_x, hmap_y, hmap_z

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
        nonlocal is_playing
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
