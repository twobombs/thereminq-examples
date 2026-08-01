# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_k4.py
# K4 All-Boundary Manifold Cluster Annealing - Visualization Dashboard
# Reads output written by K4ManifoldEngine (Rev 301-303).
#
# Input files (written by K4ManifoldEngine):
#   k4_manifold_states.npy                 shape (T, 4, 27, 3) float32
#   k4_manifold_config.json
#   k4_exact_ground_state_energy_curve.csv
#   k4_junction_profiles.csv
#   k4_loop_momentum_spectrum.csv
#
# Layout
# ------
# Left  : K4 tetrahedron graph (3D, animated)
#   nodes    per-manifold mean <Z>, colored by spin_cmap [-1..+1]
#   edges    junction coupling magnitude |mean Bloch|, colored by plasma
#            loop junctions (J01 J12 J23 J03) solid thick
#            chord junctions (J02 J13) dashed thin
#   quivers  mean Bloch direction at each manifold node (white arrows)
#   labels   M0..M3 node labels, J## edge midpoint labels (static)
#
# Right : six stacked panels
#   [0] Energy: Total / Exact Bulk / Junction vs Trotter step
#   [1] Min Unitary Fidelity + Swap Asymmetry M0M1 and M2M3
#   [2] Loop Momentum DFT: |A_k| for k=0..3 vs Trotter step
#   [3] Heatmap <X>  4 manifolds x 27 qubits
#   [4] Heatmap <Y>
#   [5] Heatmap <Z>
#
# Controls
# --------
#   Slider       scrub to any Trotter step
#   Play/Pause   toggle animation (slow azimuth rotation in play mode)
#   Space        600 dpi PNG snapshot named by step and energy
#   Scroll       zoom K4 3D panel (Matplotlib >= 3.3 / 3.8)

import sys
import csv
import json
import math
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# FILE CONFIGURATION
# ---------------------------------------------------------------------------
DATA_FILE     = "k4_manifold_states.npy"
CONFIG_FILE   = "k4_manifold_config.json"
ENERGY_FILE   = "k4_exact_ground_state_energy_curve.csv"
PROFILES_FILE = "k4_junction_profiles.csv"
LOOP_CSV      = "k4_loop_momentum_spectrum.csv"
SAVE_FILE     = "k4_manifold_dash.mp4"

# ---------------------------------------------------------------------------
# K4 TOPOLOGY CONSTANTS   (must match Rev 303 engine globals)
# ---------------------------------------------------------------------------
N_MANIFOLDS         = 4
QUBITS_PER_MANIFOLD = 27
JUNCTION_PAIRS      = [(i, j) for i in range(N_MANIFOLDS)
                       for j in range(i + 1, N_MANIFOLDS)]   # 6 junctions
LOOP_CYCLE          = [0, 1, 2, 3]
LOOP_JUNCTIONS      = [tuple(sorted((LOOP_CYCLE[i], LOOP_CYCLE[(i + 1) % 4])))
                       for i in range(4)]
CHORD_JUNCTIONS     = [jp for jp in JUNCTION_PAIRS if jp not in LOOP_JUNCTIONS]

# Regular tetrahedron vertices on the unit circumsphere.
# All 6 inter-node distances are identical (= 2*sqrt(2/3)),
# giving the natural 3D embedding of the complete graph K4.
_T = 1.0 / math.sqrt(3.0)
MANIFOLD_POS = np.array([
    [ _T,  _T,  _T],   # M0
    [ _T, -_T, -_T],   # M1
    [-_T,  _T, -_T],   # M2
    [-_T, -_T,  _T],   # M3
], dtype=float)

# ---------------------------------------------------------------------------
# ANALYTICS LOADING
# ---------------------------------------------------------------------------

def load_k4_analytics(num_steps):
    """Load all CSV analytics written by K4ManifoldEngine.

    Returns
    -------
    energies         dict of lists length num_steps (nan-padded if partial run)
    loop_k_series    ndarray (num_steps, 4): |A_k| amplitude norm per step
    junction_profiles dict: step -> {junction_label -> np.array([X,Y,Z])}

    Assumes measure_every=1 so CSV Step equals history array index.
    """
    energies = {k: [] for k in ('Total', 'Bulk', 'Junction',
                                 'Fidelity', 'Asym_M0M1', 'Asym_M2M3')}
    loop_k_series   = np.full((num_steps, 4), np.nan)
    junction_profiles = {}

    def _f(row, col):
        try:
            return float(row.get(col, 'nan'))
        except ValueError:
            return np.nan

    try:
        with open(ENERGY_FILE, mode='r') as fh:
            for row in csv.DictReader(fh):
                energies['Total'].append(_f(row, 'Total_Energy'))
                energies['Bulk'].append(_f(row, 'Exact_Bulk_Energy'))
                energies['Junction'].append(_f(row, 'Junction_Energy'))
                energies['Fidelity'].append(_f(row, 'Min_Unitary_Fidelity'))
                energies['Asym_M0M1'].append(_f(row, 'Swap_Asym_M0M1'))
                energies['Asym_M2M3'].append(_f(row, 'Swap_Asym_M2M3'))
    except Exception as exc:
        print(f"Warning: Could not parse {ENERGY_FILE}: {exc}")

    for key in energies:
        arr = energies[key]
        if len(arr) < num_steps:
            arr.extend([np.nan] * (num_steps - len(arr)))
        energies[key] = arr[:num_steps]

    try:
        with open(LOOP_CSV, mode='r') as fh:
            for row in csv.DictReader(fh):
                s = int(row['Step'])
                k = int(row['Loop_Momentum'])
                if 0 <= s < num_steps and 0 <= k < 4:
                    try:
                        norm = math.sqrt(float(row['Amp_X'])**2
                                         + float(row['Amp_Y'])**2
                                         + float(row['Amp_Z'])**2)
                    except (ValueError, KeyError):
                        norm = np.nan
                    loop_k_series[s, k] = norm
    except Exception as exc:
        print(f"Warning: Could not parse {LOOP_CSV}: {exc}")

    try:
        with open(PROFILES_FILE, mode='r') as fh:
            for row in csv.DictReader(fh):
                s     = int(row['Step'])
                label = row['Junction']
                if s not in junction_profiles:
                    junction_profiles[s] = {}
                junction_profiles[s][label] = np.array([
                    float(row['X_mean']),
                    float(row['Y_mean']),
                    float(row['Z_mean']),
                ])
    except Exception as exc:
        print(f"Warning: Could not parse {PROFILES_FILE}: {exc}")

    return energies, loop_k_series, junction_profiles


# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------

def run_dashboard(mode="interactive"):
    prefix = "[BG Render] " if mode == "save" else "[Viewer] "

    # Load config
    try:
        with open(CONFIG_FILE, 'r') as fh:
            config = json.load(fh)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    twist        = config.get("twist_gluing", True)
    gluing_label = "Moebius" if twist else "Trivial"

    # Load state history
    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    if (history.ndim != 4
            or history.shape[1] != N_MANIFOLDS
            or history.shape[2] != QUBITS_PER_MANIFOLD):
        print(f"{prefix}Error: unexpected history shape {history.shape}. "
              f"Expected (T, {N_MANIFOLDS}, {QUBITS_PER_MANIFOLD}, 3).")
        sys.exit(1)

    num_steps = history.shape[0]
    print(f"{prefix}Loaded {num_steps} steps, shape {history.shape}.")

    # Load analytics
    energies, loop_k_series, junction_profiles = load_k4_analytics(num_steps)

    # -----------------------------------------------------------------------
    # Colormaps and norms
    # -----------------------------------------------------------------------
    spin_colors = [
        (0.15, 0.35, 0.85, 1.0),   # -1 = blue  (spin-down)
        (0.08, 0.08, 0.08, 1.0),   #  0 = near-black (equatorial)
        (0.85, 0.15, 0.25, 1.0),   # +1 = red   (spin-up)
    ]
    spin_cmap = mcolors.LinearSegmentedColormap.from_list("spin_k4", spin_colors)
    spin_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    # Junction edge coupling: magnitude normalised to [0, 1] where 1 = sqrt(3)
    edge_cmap = plt.cm.plasma
    edge_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Loop-momentum mode colors: k=0 gold, k=1 cyan, k=2 salmon, k=3 green
    k_colors = ['#ffd700', '#00e5ff', '#ff6b6b', '#90ee90']

    # -----------------------------------------------------------------------
    # Figure and GridSpec layout
    # -----------------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(21, 11))
    gs  = gridspec.GridSpec(
        1, 2, width_ratios=[2.2, 1], wspace=0.06,
        left=0.04, right=0.97, top=0.94, bottom=0.13
    )

    ax3d      = fig.add_subplot(gs[0], projection='3d')
    gs_right  = gridspec.GridSpecFromSubplotSpec(
        6, 1, subplot_spec=gs[1], hspace=0.90
    )
    ax_energy = fig.add_subplot(gs_right[0])
    ax_fid    = fig.add_subplot(gs_right[1])
    ax_loop   = fig.add_subplot(gs_right[2])
    ax_hmap_x = fig.add_subplot(gs_right[3])
    ax_hmap_y = fig.add_subplot(gs_right[4])
    ax_hmap_z = fig.add_subplot(gs_right[5])

    # -----------------------------------------------------------------------
    # Static 3D setup: axes, labels, edge lines
    # -----------------------------------------------------------------------
    ax3d.set_xlim(-1.1, 1.1)
    ax3d.set_ylim(-1.1, 1.1)
    ax3d.set_zlim(-1.1, 1.1)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.line.set_linewidth(0)
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()
    try:
        ax3d.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    # Static manifold node labels (offset outward from tetrahedron center)
    _label_offset = 0.22
    for m in range(N_MANIFOLDS):
        n_dir = MANIFOLD_POS[m] / (np.linalg.norm(MANIFOLD_POS[m]) + 1e-9)
        lx = MANIFOLD_POS[m, 0] + n_dir[0] * _label_offset
        ly = MANIFOLD_POS[m, 1] + n_dir[1] * _label_offset
        lz = MANIFOLD_POS[m, 2] + n_dir[2] * _label_offset
        ax3d.text(lx, ly, lz, f"M{m}",
                  color='white', fontsize=12, fontweight='bold',
                  ha='center', va='center')

    # Static junction edge lines and midpoint labels.
    # Loop junctions: solid/thick; chord junctions: dashed/thin.
    edge_lines = []
    for ji, (ma, mb) in enumerate(JUNCTION_PAIRS):
        pa = MANIFOLD_POS[ma]
        pb = MANIFOLD_POS[mb]
        is_loop = (ma, mb) in LOOP_JUNCTIONS
        ls = '-'   if is_loop else '--'
        lw = 2.2   if is_loop else 1.2
        line, = ax3d.plot(
            [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
            color='#333333', linewidth=lw, linestyle=ls, zorder=1
        )
        edge_lines.append(line)
        mid  = (pa + pb) * 0.5
        n_m  = mid / (np.linalg.norm(mid) + 1e-9)
        kind = "L" if is_loop else "C"
        ax3d.text(
            mid[0] + n_m[0] * 0.12,
            mid[1] + n_m[1] * 0.12,
            mid[2] + n_m[2] * 0.12,
            f"J{ma}{mb}({kind})",
            color='#777777', fontsize=7, ha='center', va='center'
        )

    # Legend annotation for edge style
    ax3d.text2D(0.02, 0.04,
                "L = loop junction (solid)   C = chord junction (dashed)",
                transform=ax3d.transAxes, fontsize=8, color='#888888')

    # Container for per-frame dynamic 3D artists (scatter + quivers)
    k4_dynamic = []

    # -----------------------------------------------------------------------
    # Static time-series panels
    # -----------------------------------------------------------------------
    x_axis = np.arange(num_steps)

    # [0] Energy
    ax_energy.plot(x_axis, energies['Total'],    color='lightgreen',
                   label='Total',   linewidth=1.5)
    ax_energy.plot(x_axis, energies['Bulk'],     color='dodgerblue',
                   label='Bulk',    linewidth=1.2)
    ax_energy.plot(x_axis, energies['Junction'], color='#ff9933',
                   label='Junction', linewidth=1.2)
    ax_energy.set_title("Energy Components", fontsize=9)
    ax_energy.legend(fontsize=7, loc='upper right')
    ax_energy.grid(True, alpha=0.2)
    ax_energy.set_xticks([])
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # [1] Fidelity + Swap Asymmetry (twin y-axes)
    ax_fid.plot(x_axis, energies['Fidelity'],
                color='cyan', label='Min Fidelity', linewidth=1.5)
    ax_fid.set_ylim(0.0, 1.05)
    ax_fid.set_title("Fidelity + Swap Asymmetry", fontsize=9)
    ax_fid.legend(fontsize=7, loc='upper left')
    ax_fid.grid(True, alpha=0.2)
    ax_fid.set_xticks([])
    ax_fid_r = ax_fid.twinx()
    ax_fid_r.plot(x_axis, energies['Asym_M0M1'],
                  color='#ff9900', label='Asym M0M1', alpha=0.85, linewidth=1.2)
    ax_fid_r.plot(x_axis, energies['Asym_M2M3'],
                  color='#ff44aa', label='Asym M2M3', alpha=0.85, linewidth=1.2)
    ax_fid_r.set_ylim(0.0, None)
    ax_fid_r.legend(fontsize=7, loc='upper right')
    vline_f = ax_fid.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # [2] Loop Momentum DFT spectrum
    k_labels = ['k=0 (uniform)', 'k=1 (ringing)', 'k=2', 'k=3']
    for k in range(4):
        ax_loop.plot(x_axis, loop_k_series[:, k],
                     color=k_colors[k], label=k_labels[k], linewidth=1.5)
    ax_loop.set_title("Loop Momentum DFT  |A_k|", fontsize=9)
    ax_loop.legend(fontsize=7, loc='upper right')
    ax_loop.set_ylim(0.0, None)
    ax_loop.grid(True, alpha=0.2)
    ax_loop.set_xticks([])
    vline_loop = ax_loop.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # [3-5] Per-manifold qubit heatmaps  (4 manifolds x 27 qubits)
    manifold_yticks  = list(range(N_MANIFOLDS))
    manifold_ylabels = [f"M{m}" for m in manifold_yticks]

    def _init_heatmap(ax, data, title, show_xlabel=False):
        img = ax.imshow(data, cmap=spin_cmap, norm=spin_norm,
                        aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_yticks(manifold_yticks)
        ax.set_yticklabels(manifold_ylabels, fontsize=8)
        # Thin separator lines between manifolds
        for m in range(1, N_MANIFOLDS):
            ax.axhline(m - 0.5, color='#555555', linewidth=0.8)
        # Vertical separator every 9 qubits (one intra-manifold edge group)
        for q in range(9, QUBITS_PER_MANIFOLD, 9):
            ax.axvline(q - 0.5, color='#444444', linewidth=0.8, linestyle=':')
        if show_xlabel:
            ax.set_xlabel("Qubit Index (0-26)  |  groups: 0-8 / 9-17 / 18-26",
                          fontsize=7)
            ax.tick_params(axis='x', labelsize=6)
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0],
                            "Polarization <X>")
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1],
                            "Polarization <Y>")
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2],
                            "Polarization <Z>", show_xlabel=True)

    # Shared colorbar (left margin, spanning roughly the heatmap rows)
    ax_cbar = fig.add_axes([0.005, 0.22, 0.012, 0.52])
    sm = plt.cm.ScalarMappable(cmap=spin_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('<Z> Spin', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # -----------------------------------------------------------------------
    # Slider and Play/Pause button
    # -----------------------------------------------------------------------
    ax_slider = fig.add_axes([0.10, 0.05, 0.65, 0.025])
    slider = Slider(
        ax=ax_slider, label='Step',
        valmin=0, valmax=num_steps - 1,
        valinit=0, valstep=1, color='#4a90e2'
    )

    ax_play = fig.add_axes([0.80, 0.038, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')
    is_playing = [True]

    # -----------------------------------------------------------------------
    # Scroll-wheel zoom
    # -----------------------------------------------------------------------
    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        if not hasattr(ax3d, '_k4_zoom'):
            ax3d._k4_zoom = 1.0
        ax3d._k4_zoom *= 0.9 if event.button == 'down' else 1.1
        try:
            ax3d.set_box_aspect((1, 1, 1), zoom=ax3d._k4_zoom)
        except TypeError:
            ax3d.dist *= 0.9 if event.button == 'down' else 1.1
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # -----------------------------------------------------------------------
    # Spacebar snapshot
    # -----------------------------------------------------------------------
    def on_key_press(event):
        if event.key != ' ':
            return
        step  = int(slider.val)
        e_val = energies['Total'][step]
        e_str = f"{e_val:.4f}" if not (isinstance(e_val, float) and
                                        math.isnan(e_val)) else "NaN"
        fname = f"k4_snap_step{step:03d}_E{e_str}.png"
        print(f"{prefix}Saving screenshot -> {fname}")
        fig.savefig(fname, dpi=600, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"{prefix}Screenshot saved.")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # -----------------------------------------------------------------------
    # Per-frame update function
    # -----------------------------------------------------------------------
    _from_animation = [False]

    def update(frame):
        frame = int(frame)

        # Remove previous dynamic 3D artists
        for artist in k4_dynamic:
            try:
                artist.remove()
            except Exception:
                pass
        k4_dynamic.clear()

        # Per-manifold statistics
        mean_z     = history[frame, :, :, 2].mean(axis=1)   # shape (4,)
        mean_bloch = history[frame, :, :, :].mean(axis=1)   # shape (4, 3)

        # Manifold node scatter colored by mean <Z>
        sc = ax3d.scatter(
            MANIFOLD_POS[:, 0], MANIFOLD_POS[:, 1], MANIFOLD_POS[:, 2],
            c=mean_z, cmap=spin_cmap, norm=spin_norm,
            s=420, depthshade=False, zorder=5, edgecolors='white',
            linewidths=0.8
        )
        k4_dynamic.append(sc)

        # Mean Bloch quiver arrows at each manifold node
        _scale = 0.40
        for m in range(N_MANIFOLDS):
            bvec = mean_bloch[m] * _scale
            q = ax3d.quiver(
                MANIFOLD_POS[m, 0], MANIFOLD_POS[m, 1], MANIFOLD_POS[m, 2],
                bvec[0], bvec[1], bvec[2],
                color='white', linewidth=1.8, arrow_length_ratio=0.35
            )
            k4_dynamic.append(q)

        # Junction edge colors from coupling magnitude
        jp_data = junction_profiles.get(frame, {})
        _sqrt3  = math.sqrt(3.0)
        for ji, (ma, mb) in enumerate(JUNCTION_PAIRS):
            label = f"J{ma}{mb}"
            vec   = jp_data.get(label, np.zeros(3))
            mag   = float(np.linalg.norm(vec))
            mag_n = min(mag / _sqrt3, 1.0)           # normalise to [0, 1]
            color = edge_cmap(edge_norm(mag_n))
            edge_lines[ji].set_color(color)
            edge_lines[ji].set_alpha(0.40 + 0.60 * mag_n)

        # Heatmaps
        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])

        # Vertical step markers on all time-series panels
        for vline in (vline_e, vline_f, vline_loop):
            vline.set_xdata([frame, frame])

        # 3D panel title
        e_val = energies['Total'][frame]
        e_str = (f"{e_val:+.4f}"
                 if not (isinstance(e_val, float) and math.isnan(e_val))
                 else "?")
        fid   = energies['Fidelity'][frame]
        f_str = (f"{fid:.4f}"
                 if not (isinstance(fid, float) and math.isnan(fid))
                 else "?")
        ax3d.set_title(
            f"K4 Manifold Cluster  |  {gluing_label} Gluing  |  "
            f"Step {frame}/{num_steps - 1}  |  E={e_str}  |  Fid={f_str}",
            fontsize=11, pad=8
        )

        # Slow azimuth rotation while animation is running
        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.40)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return tuple(k4_dynamic) + (hmap_x, hmap_y, hmap_z)

    def _anim_update(frame):
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
        if is_playing[0]:
            ani.event_source.stop()
            btn_play.label.set_text('Play')
        else:
            ani.event_source.start()
            btn_play.label.set_text('Pause')
        is_playing[0] = not is_playing[0]
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    # blit=False required: Axes3D does not support blitting.
    ani = animation.FuncAnimation(
        fig, _anim_update, frames=num_steps, interval=150, blit=False
    )

    if mode == "save":
        print(f"{prefix}Starting 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Render complete.")
        except Exception as exc:
            print(f"{prefix}Render failed (ffmpeg installed?): {exc}")
    else:
        print(f"{prefix}Opening GUI...")
        plt.show()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    mp.set_start_method('spawn', force=True)
    print("Forking 4K render to background process...")
    render_proc = mp.Process(target=run_dashboard, args=("save",))
    render_proc.start()

    run_dashboard(mode="interactive")

    if render_proc.is_alive():
        print("\nInteractive viewer closed. Waiting for background render...")
        render_proc.join()
    print("All processes terminated.")


if __name__ == "__main__":
    main()
