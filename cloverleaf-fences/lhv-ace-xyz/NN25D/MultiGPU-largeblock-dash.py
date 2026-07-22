# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_blocks_v10.py
# Dashboard for Rev 88-M+ Block-Lattice engine (256 patches, 4096 qubits).
#
# PORTED FROM cloverfield rubics dash (Rev 120):
#   - Min_Unitary_Fidelity overlay on energy panel (purple twin axis)
#   - Ring_Reset events panel: per-step bar chart + vlines on all panels
#   - Anneal schedule panel: Anneal_Percent curve
#   - Status bar: step / anneal % / fidelity / ring-reset count live readout
#   - Panel style: #1a1a1a face, #333333 spines, tick color #aaaaaa throughout
#   - Energy panel now uses proper dark-panel draw_energy() function style
#   - Convergence (derivatives) panel rewritten to match cloverfield pattern:
#     dE/dt left axis (green), d||ds||/dt right axis (crimson), both from
#     valid data only via _safe_gradient
#   - Ring_Reset vlines on energy, coupling-error, convergence, anneal panels
#
# KEPT FROM largeblock v9:
#   - Full block-lattice 3D quiver (4096 arrows, QUIVER_STRIDE subsampling)
#   - Inter-block Z translucent plane sheets (Z_INTER visual cue)
#   - Per-kind coupling-error panel (Z_INTRA / Z_INTER / XY lines)
#   - Config-driven geometry from lattice_config.json
#   - Background 4K FFmpeg render process (dual-process model)
#   - Heatmap tile-column / block separators (per-tile labels)
#   - Scroll-to-zoom on 3D panel, SPACE = PNG snapshot
#
# RIGHT PANEL LAYOUT (7 rows):
#   Row 0 : Energy components + fidelity twin axis
#   Row 1 : Ring_Reset events (bar + count)
#   Row 2 : Coupling error by seam class (Z_INTRA / Z_INTER / XY)
#   Row 3 : Anneal schedule + convergence derivatives (twin)
#   Row 4 : <X> heatmap
#   Row 5 : <Y> heatmap
#   Row 6 : <Z> heatmap

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
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- CONFIGURATION ---
DATA_FILE   = "macroscopic_lattice_states.npy"
CONFIG_FILE = "lattice_config.json"
ENERGY_FILE = "meanfield_ground_state_energy_curve_multi.csv"
SAVE_FILE   = "macroscopic_lattice_dash_blocks.mp4"

TILE_LX, TILE_LY = 4, 4
LAYER_SPACING    = 1.6
BLOCK_GAP_Z      = 1.2
TILE_GAP_XY      = 1.5
QUIVER_STRIDE    = 1        # set >1 to speed up interactive 3D

KIND_COLORS = {"Z_INTRA": "#e6b422", "Z_INTER": "#e64550", "XY": "#45b0e6"}

# Colour constants matching cloverfield style
RING_RESET_COLOR = "#ff4444"
FIDELITY_COLOR   = "#cc88ff"
ANNEAL_COLOR     = "#f5c518"
# ---------------------


# =====================================================================
# DATA LOADING
# =====================================================================

def load_energy_data(num_steps, log_prefix=""):
    """Load all columns from the energy CSV, including Rev 88-D+ additions."""
    energies = {
        'Total': [], 'Bulk': [], 'Boundary': [],
        'Z_INTRA': [], 'Z_INTER': [], 'XY': [],
        'Fidelity': [], 'Anneal': [], 'Ring_Reset': [],
    }
    kind_cols = {'Z_INTRA': "E_Z_Intra", 'Z_INTER': "E_Z_Inter", 'XY': "E_XY"}

    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                def _get(col, default=np.nan):
                    try:
                        return float(row[col])
                    except (KeyError, ValueError):
                        return default

                energies['Total'].append(_get("MeanField_Total_Energy"))
                energies['Bulk'].append(_get("MeanField_Bulk_Energy"))
                energies['Boundary'].append(_get("MeanField_Boundary_Energy"))
                energies['Fidelity'].append(_get("Min_Unitary_Fidelity"))
                energies['Anneal'].append(_get("Anneal_Percent"))
                energies['Ring_Reset'].append(_get("Ring_Reset", 0.0))
                for kind, col in kind_cols.items():
                    energies[kind].append(_get(col))
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE}: {e}")

    for key in energies:
        arr = energies[key]
        if len(arr) < num_steps:
            arr.extend([np.nan] * (num_steps - len(arr)))
        energies[key] = np.array(arr[:num_steps], dtype=float)

    return energies


def _safe_gradient(arr_with_nans):
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 1:
        out[valid_indices] = np.gradient(arr[valid_indices], valid_indices)
    elif len(valid_indices) == 1:
        out[valid_indices[0]] = 0.0
    return out


# =====================================================================
# INTERFACE GEOMETRY (mirrors engine build_interfaces exactly)
# =====================================================================

def build_interfaces(grid_x, grid_y, grid_z, branes_per_block, qpp,
                     periodic_x, periodic_y, periodic_z):
    def patch_id(tx, ty, z):
        return (tx * grid_y + ty) * grid_z + z

    z_i1 = np.arange(qpp)
    z_i2 = np.arange(qpp)
    x_i1 = np.array([12 + y for y in range(4)])
    x_i2 = np.array([y for y in range(4)])
    y_i1 = np.array([x * 4 + 3 for x in range(4)])
    y_i2 = np.array([x * 4 for x in range(4)])

    interfaces = []
    for tx in range(grid_x):
        for ty in range(grid_y):
            for z in range(grid_z):
                p1 = patch_id(tx, ty, z)
                if z < grid_z - 1:
                    kind = "Z_INTRA" if (z + 1) % branes_per_block != 0 else "Z_INTER"
                    interfaces.append((p1, patch_id(tx, ty, z + 1), z_i1, z_i2, kind))
                elif periodic_z and grid_z > 2:
                    interfaces.append((p1, patch_id(tx, ty, 0), z_i1, z_i2, "Z_INTER"))
                if tx < grid_x - 1:
                    interfaces.append((p1, patch_id(tx + 1, ty, z), x_i1, x_i2, "XY"))
                elif periodic_x and grid_x > 2:
                    interfaces.append((p1, patch_id(0, ty, z), x_i1, x_i2, "XY"))
                if ty < grid_y - 1:
                    interfaces.append((p1, patch_id(tx, ty + 1, z), y_i1, y_i2, "XY"))
                elif periodic_y and grid_y > 2:
                    interfaces.append((p1, patch_id(tx, 0, z), y_i1, y_i2, "XY"))
    return interfaces


# =====================================================================
# PANEL DRAW HELPERS (cloverfield-style: clear+redraw per frame)
# =====================================================================

def _style_ax(ax):
    """Apply cloverfield panel style: dark face, dim spines, grey ticks."""
    ax.set_facecolor('#1a1a1a')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')
    ax.tick_params(colors='#aaaaaa', labelsize=6)


def _ring_reset_vlines(ax, ring_reset_arr, alpha=0.18):
    """Draw faint red axvspans on steps where Ring_Reset==1."""
    for i, v in enumerate(ring_reset_arr):
        if v == 1.0:
            ax.axvspan(i - 0.5, i + 0.5, color=RING_RESET_COLOR,
                       alpha=alpha, linewidth=0)


def draw_energy_panel(ax, energies, step_cursor, num_steps):
    ax.cla(); _style_ax(ax)
    xs = np.arange(num_steps)

    _ring_reset_vlines(ax, energies['Ring_Reset'])

    def _plot(col, color, lw, label, ls='-'):
        arr = energies[col]
        if not np.all(np.isnan(arr)):
            ax.plot(xs, arr, color=color, linewidth=lw,
                    linestyle=ls, label=label)

    _plot('Total',    'lightgreen',  1.2, 'Total')
    _plot('Bulk',     'dodgerblue',  1.0, 'Bulk')
    _plot('Boundary', 'orange',      1.0, 'Bndry (all)')
    for k, lbl in (("Z_INTRA", "Zi"), ("Z_INTER", "Ze"), ("XY", "XY")):
        if not np.all(np.isnan(energies[k])):
            ax.plot(xs, energies[k], color=KIND_COLORS[k],
                    linewidth=0.8, linestyle='--', alpha=0.9, label=lbl)

    # Fidelity on twin axis
    fid = energies['Fidelity']
    if not np.all(np.isnan(fid)):
        ax2 = ax.twinx()
        ax2.plot(xs, fid, color=FIDELITY_COLOR, linewidth=0.8,
                 linestyle=':', label='Fidelity')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Fidelity", fontsize=6, color=FIDELITY_COLOR)
        ax2.tick_params(colors=FIDELITY_COLOR, labelsize=5)
        ax2.set_facecolor('#1a1a1a')
        for sp in ax2.spines.values():
            sp.set_edgecolor('#333333')

    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_title("Energy components + fidelity", fontsize=9, color='#cccccc', pad=3)
    ax.set_ylabel("Energy (a.u.)", fontsize=7, color='#aaaaaa')
    ax.legend(fontsize=5, loc='upper left', ncol=3,
              facecolor='#1a1a1a', labelcolor='#cccccc', framealpha=0.5)
    ax.grid(True, alpha=0.15)


def draw_ring_reset_panel(ax, energies, step_cursor, num_steps):
    ax.cla(); _style_ax(ax)
    xs = np.arange(num_steps)
    rr = energies['Ring_Reset']

    # Bar chart: 1 = reset, 0 = clean
    ax.bar(xs, rr, color=RING_RESET_COLOR, width=1.0, alpha=0.75)
    # Cumulative count on twin axis
    cumsum = np.nancumsum(np.where(np.isnan(rr), 0, rr))
    ax2 = ax.twinx()
    ax2.plot(xs, cumsum, color='#ffaa44', linewidth=0.9,
             linestyle='-', label='Cumulative resets')
    ax2.set_ylabel("Cumul. resets", fontsize=6, color='#ffaa44')
    ax2.tick_params(colors='#ffaa44', labelsize=5)
    ax2.set_facecolor('#1a1a1a')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#333333')

    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
    total_rr = int(np.nansum(rr))
    ax.set_title(f"Ring Reset events  (total: {total_rr})",
                 fontsize=9, color='#cccccc', pad=3)
    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ok', 'RESET'], fontsize=5, color='#aaaaaa')
    ax.grid(True, alpha=0.1)


def draw_coupling_error_panel(ax, avg_disagreement, dis_by_kind,
                               energies, step_cursor, num_steps):
    ax.cla(); _style_ax(ax)
    if avg_disagreement is None:
        ax.text(0.5, 0.5, "No interface data",
                ha='center', va='center', transform=ax.transAxes,
                color='#555555', fontsize=9)
        ax.set_title("Coupling error", fontsize=9, color='#cccccc', pad=3)
        return

    xs = np.arange(num_steps)
    _ring_reset_vlines(ax, energies['Ring_Reset'])

    ax.plot(xs, avg_disagreement, color='crimson', linewidth=1.4,
            label='Mean (all seams)')
    for k, lbl in (("Z_INTRA", "Z-intra"), ("Z_INTER", "Z-inter"), ("XY", "XY")):
        if k in dis_by_kind:
            ax.plot(xs, dis_by_kind[k], color=KIND_COLORS[k],
                    linewidth=0.8, alpha=0.9, label=lbl)

    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_title("Coupling error by seam class\n"
                 "mean_seams( mean_k ||s_{i1} - s_{i2}|| )",
                 fontsize=8, color='#cccccc', pad=3)
    ax.set_ylabel("Error (L2)", fontsize=7, color='#aaaaaa')
    ax.legend(fontsize=5, loc='upper left', ncol=2,
              facecolor='#1a1a1a', labelcolor='#cccccc', framealpha=0.5)
    ax.grid(True, alpha=0.15)


def draw_anneal_deriv_panel(ax, energies, avg_disagreement,
                             step_cursor, num_steps):
    """Anneal schedule (left) + dE/dt and d||ds||/dt (right twin)."""
    ax.cla(); _style_ax(ax)
    xs = np.arange(num_steps)

    _ring_reset_vlines(ax, energies['Ring_Reset'])

    # Anneal % on left axis
    ann = energies['Anneal']
    if not np.all(np.isnan(ann)):
        ax.plot(xs, ann, color=ANNEAL_COLOR, linewidth=1.0,
                label='Anneal %')
    ax.set_ylabel("Anneal %", fontsize=7, color=ANNEAL_COLOR)
    ax.tick_params(axis='y', colors=ANNEAL_COLOR, labelsize=5)

    # dE/dt and d||ds||/dt on right twin axis
    ax2 = ax.twinx()
    dE = _safe_gradient(energies['Total'])
    ax2.plot(xs, dE, color='lightgreen', linewidth=0.85,
             linestyle='-', label='dE/dt')
    if avg_disagreement is not None:
        dR = _safe_gradient(avg_disagreement)
        ax2.plot(xs, dR, color='crimson', linewidth=0.85,
                 linestyle='--', label='d||ds||/dt')
    ax2.set_ylabel("Rate (per step)", fontsize=6, color='#aaaaaa')
    ax2.tick_params(colors='#aaaaaa', labelsize=5)
    ax2.set_facecolor('#1a1a1a')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#333333')

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5, loc='upper left',
              facecolor='#1a1a1a', labelcolor='#cccccc', framealpha=0.5)

    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_title("Anneal schedule + convergence rates", fontsize=9,
                 color='#cccccc', pad=3)
    ax.grid(True, alpha=0.15)


# =====================================================================
# MAIN DASHBOARD
# =====================================================================

def run_dashboard(mode="interactive"):
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "

    # --- Config ---
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    grid_x           = config.get("grid_x", 4)
    grid_y           = config.get("grid_y", 4)
    grid_z           = config.get("grid_z", 16)
    block_grid       = config.get("block_grid", [4, 4, 4])
    branes_per_block = config.get("branes_per_block", 4)
    cfg_qpp          = config.get("qubits_per_patch", TILE_LX * TILE_LY)
    tile_geometry    = config.get("tile_geometry", "4x4_brane_block_lattice")
    periodic         = config.get("periodic", [False, False, False])
    if not isinstance(periodic, (list, tuple)):
        periodic = [bool(periodic)] * 3
    periodic_x, periodic_y, periodic_z = (bool(v) for v in periodic)

    if tile_geometry != "4x4_brane_block_lattice":
        print(f"{prefix}Warning: tile_geometry='{tile_geometry}'")

    # --- State history ---
    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps  = history.shape[0]
    num_patches = history.shape[1]
    qpp        = history.shape[2]
    total_qubits = num_patches * qpp

    def patch_coords(p):
        z = p % grid_z; rest = p // grid_z
        return rest // grid_y, rest % grid_y, z

    # --- Energy data (all columns) ---
    energies = load_energy_data(num_steps, prefix)

    # --- Interfaces & coupling error ---
    interfaces = build_interfaces(grid_x, grid_y, grid_z, branes_per_block,
                                  qpp, periodic_x, periodic_y, periodic_z)
    kinds_present = ["Z_INTRA", "Z_INTER", "XY"]
    n_by_kind = {k: sum(1 for i in interfaces if i[4] == k) for k in kinds_present}
    print(f"{prefix}Interfaces: {n_by_kind['Z_INTRA']} Z-intra, "
          f"{n_by_kind['Z_INTER']} Z-inter, {n_by_kind['XY']} XY")

    avg_disagreement = None
    dis_by_kind = {}
    if interfaces:
        hist_arr = np.asarray(history)
        n_ifaces = len(interfaces)
        disagreements = np.zeros((num_steps, n_ifaces))
        iface_kind = []
        for i, (p1, p2, i1, i2, kind) in enumerate(interfaces):
            diff = hist_arr[:, p1, i1, :] - hist_arr[:, p2, i2, :]
            disagreements[:, i] = np.mean(np.linalg.norm(diff, axis=2), axis=1)
            iface_kind.append(kind)
        iface_kind = np.array(iface_kind)
        avg_disagreement = np.mean(disagreements, axis=1)
        for k in kinds_present:
            mask = (iface_kind == k)
            if np.any(mask):
                dis_by_kind[k] = np.mean(disagreements[:, mask], axis=1)

    # --- 3D coordinates ---
    tile_pitch_x = TILE_LX - 1 + TILE_GAP_XY + 1
    tile_pitch_y = TILE_LY - 1 + TILE_GAP_XY + 1

    def z_pos(z):
        return z * LAYER_SPACING + (z // branes_per_block) * BLOCK_GAP_Z

    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        tx, ty, z = patch_coords(p)
        zp = z_pos(z)
        for q in range(qpp):
            qx, qy = divmod(q, TILE_LY)
            global_X.append(tx * tile_pitch_x + float(qx))
            global_Y.append(ty * tile_pitch_y + float(qy))
            global_Z.append(zp)
    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    x_max = (grid_x - 1) * tile_pitch_x + TILE_LX - 1
    y_max = (grid_y - 1) * tile_pitch_y + TILE_LY - 1
    z_max = z_pos(grid_z - 1)

    stride   = max(1, int(QUIVER_STRIDE))
    draw_idx = np.arange(0, num_patches * qpp, stride)
    qX, qY, qZ = global_X[draw_idx], global_Y[draw_idx], global_Z[draw_idx]

    # --- Figure layout ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 11), facecolor='#111111')

    # Status bar at top
    status_ax = fig.add_axes([0.03, 0.965, 0.94, 0.018])
    status_ax.set_axis_off()
    status_txt = status_ax.text(
        0.0, 0.5, "", transform=status_ax.transAxes,
        fontsize=8, color='#aaaaaa', va='center', family='monospace')

    gs_main = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1],
                                wspace=0.08, left=0.03, right=0.97,
                                top=0.955, bottom=0.13)
    ax3d = fig.add_subplot(gs_main[0], projection='3d')

    # Right column: 7 rows
    gs_right = gridspec.GridSpecFromSubplotSpec(
        7, 1, subplot_spec=gs_main[1], hspace=0.90,
        height_ratios=[1.1, 0.55, 0.85, 0.75, 0.75, 0.75, 0.75])
    ax_energy  = fig.add_subplot(gs_right[0])
    ax_rr      = fig.add_subplot(gs_right[1])
    ax_dis     = fig.add_subplot(gs_right[2])
    ax_anneal  = fig.add_subplot(gs_right[3])
    ax_hmap_x  = fig.add_subplot(gs_right[4])
    ax_hmap_y  = fig.add_subplot(gs_right[5])
    ax_hmap_z  = fig.add_subplot(gs_right[6])

    # --- Colormaps ---
    spin_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    vector_colors = [
        (0.15, 0.35, 0.85, 0.85),
        (0.85, 0.85, 0.85, 0.45),
        (0.85, 0.15, 0.25, 0.85),
    ]
    vector_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ghost_vectors", vector_colors)
    heatmap_colors = [
        (0.15, 0.35, 0.85, 1.0),
        (0.10, 0.10, 0.10, 1.0),
        (0.85, 0.15, 0.25, 1.0),
    ]
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap_cmap", heatmap_colors)

    def _quiver_colors(w_flat):
        return vector_cmap(spin_norm(w_flat))

    def get_vector_data(step_idx):
        return (
            history[step_idx, :, :, 0].ravel()[draw_idx],
            history[step_idx, :, :, 1].ravel()[draw_idx],
            history[step_idx, :, :, 2].ravel()[draw_idx],
        )

    U, V, W = get_vector_data(0)
    quiver_obj = [ax3d.quiver(
        qX, qY, qZ, U, V, W,
        length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    # Inter-block Z planes: draw only at Z_INTER seams (the weak-glue
    # inter-block boundaries). Z_INTRA and XY seams are counted in the
    # error metric but not drawn -- 624 sheets would be visual noise.
    margin = 0.4
    for zb in range(1, grid_z // branes_per_block):
        z_lo  = z_pos(zb * branes_per_block - 1)
        z_hi  = z_pos(zb * branes_per_block)
        z_mid = 0.5 * (z_lo + z_hi)
        verts = [[(-margin, -margin, z_mid),
                  (x_max + margin, -margin, z_mid),
                  (x_max + margin, y_max + margin, z_mid),
                  (-margin, y_max + margin, z_mid)]]
        plane = Poly3DCollection(verts, alpha=0.06, facecolor='#e64550',
                                 edgecolor='#e64550', linewidths=0.5)
        ax3d.add_collection3d(plane)

    # Colorbar
    ax_cbar = fig.add_axes([0.01, 0.25, 0.012, 0.50])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Bloch vector component', fontsize=8, color='#aaaaaa')
    cbar.ax.tick_params(labelsize=6, colors='#aaaaaa')

    energy_text = ax3d.text2D(
        0.04, 0.96, "", transform=ax3d.transAxes,
        color='lightgreen', fontsize=11, fontweight='bold')

    bx, by, bz = (block_grid + [4, 4, 4])[:3] if isinstance(block_grid, list) else (4, 4, 4)
    per_flags = ''.join(a for a, on in zip('XYZ', (periodic_x, periodic_y, periodic_z)) if on)
    lattice_desc = (f"{bx}x{by}x{bz} blocks x {branes_per_block} branes"
                    + (f" | periodic {per_flags}" if per_flags else ""))

    ax3d.set_xlim(-0.5, x_max + 0.5)
    ax3d.set_ylim(-0.5, y_max + 0.5)
    ax3d.set_zlim(-0.5, z_max + 0.5)
    try:
        ax3d.set_box_aspect((x_max + 1.0, y_max + 1.0, max(1.0, z_max + 1.0)))
    except AttributeError:
        pass
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor('none')
        except AttributeError:
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis.line.set_linewidth(0)
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()

    # --- Heatmap tick setup ---
    y_ticks, y_labels = [], []
    for tx in range(grid_x):
        for ty in range(grid_y):
            base = (tx * grid_y + ty) * grid_z
            y_ticks.append(base + (grid_z - 1) / 2.0)
            y_labels.append(f"T({tx},{ty})")

    def _init_heatmap(ax, data, cmap, norm, label,
                      show_ylabel=False, show_xlabel=False):
        img = ax.imshow(data, cmap=cmap, norm=norm,
                        aspect='auto', interpolation='nearest')
        _style_ax(ax)
        ax.set_title(label, fontsize=8, color='#cccccc', pad=2)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=3.5, color='#aaaaaa')
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel(f"Local qubit (0-{qpp - 1})", fontsize=6,
                          color='#aaaaaa')
            ax.tick_params(axis='x', labelsize=5, colors='#aaaaaa')
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0],
                           heatmap_cmap, spin_norm, "Polarization <X>",
                           show_ylabel=True)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1],
                           heatmap_cmap, spin_norm, "Polarization <Y>",
                           show_ylabel=True)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2],
                           heatmap_cmap, spin_norm, "Polarization <Z>",
                           show_ylabel=True, show_xlabel=True)

    # --- Slider + Play/Pause ---
    ax_slider = fig.add_axes([0.14, 0.045, 0.62, 0.018])
    ax_slider.set_facecolor('none')
    slider = Slider(ax=ax_slider, label='Trotter Step',
                    valmin=0, valmax=num_steps - 1,
                    valinit=0, valstep=1, color='#4a90e2')
    slider.label.set_color('#cccccc')
    slider.valtext.set_color(ANNEAL_COLOR)

    ax_play = fig.add_axes([0.80, 0.032, 0.08, 0.038])
    btn_play = Button(ax_play, '> Play', color='#333333', hovercolor='#555555')
    btn_play.label.set_color('#cccccc')

    is_playing = [False]
    cur_frame  = [0]

    # --- Scroll zoom on 3D ---
    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        if not hasattr(ax3d, 'custom_zoom'):
            ax3d.custom_zoom = 1.0
        ax3d.custom_zoom *= 0.9 if event.button == 'down' else 1.1
        try:
            ca = ax3d.get_box_aspect()
            if ca is None:
                ca = (x_max + 1.0, y_max + 1.0, max(1.0, z_max + 1.0))
            ax3d.set_box_aspect(ca, zoom=ax3d.custom_zoom)
        except TypeError:
            ax3d.dist *= 1.1 if event.button == 'down' else 0.9
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # --- Space = screenshot ---
    def on_key_press(event):
        if event.key == ' ':
            step = int(slider.val)
            e_val = energies['Total'][step]
            e_str = f"{e_val:.4f}" if not np.isnan(e_val) else "NaN"
            fname = (f"dash_snapshot_blocks_{bx}x{by}x{bz}x{branes_per_block}"
                     f"_step{step}_E{e_str}.png")
            print(f"{prefix}Saving screenshot to {fname}...")
            fig.savefig(fname, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"{prefix}Screenshot saved.")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # --- Full redraw function ---
    _from_animation = [False]

    def update(frame):
        frame = int(frame)
        cur_frame[0] = frame

        # 3D quiver
        U, V, W = get_vector_data(frame)
        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            qX, qY, qZ, U, V, W,
            length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3)
        ax3d.set_title(
            f"Block-Lattice Annealing  ({lattice_desc})\n"
            f"{num_patches} branes | {total_qubits} qubits | "
            f"Step {frame}/{num_steps - 1}",
            fontsize=12, pad=8, color='#f0f0f0')

        e_val = energies['Total'][frame]
        energy_text.set_text(
            f"E = {e_val:.4f}" if not np.isnan(e_val) else "")
        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)

        # Right panels (all clear+redraw)
        draw_energy_panel(ax_energy, energies, frame, num_steps)
        draw_ring_reset_panel(ax_rr, energies, frame, num_steps)
        draw_coupling_error_panel(ax_dis, avg_disagreement, dis_by_kind,
                                   energies, frame, num_steps)
        draw_anneal_deriv_panel(ax_anneal, energies, avg_disagreement,
                                frame, num_steps)

        # Heatmaps
        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])

        # Status bar
        fid_val = energies['Fidelity'][frame]
        ann_val = energies['Anneal'][frame]
        rr_total = int(np.nansum(energies['Ring_Reset'][:frame + 1]))
        fid_str = f"{fid_val:.5f}" if not np.isnan(fid_val) else "N/A"
        ann_str = f"{ann_val:.1f}%" if not np.isnan(ann_val) else "N/A"
        status_txt.set_text(
            f"Step {frame}/{num_steps - 1}  |  Anneal {ann_str}  |  "
            f"E_total = {e_val:.4f}  |  Fidelity = {fid_str}  |  "
            f"Ring resets so far: {rr_total}"
            if not np.isnan(e_val) else
            f"Step {frame}/{num_steps - 1}  |  Anneal {ann_str}  |  "
            f"Fidelity = {fid_str}  |  Ring resets so far: {rr_total}")

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return (quiver_obj[0], energy_text, hmap_x, hmap_y, hmap_z)

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
        if is_playing[0]:
            ani.event_source.stop()
            btn_play.label.set_text('> Play')
        else:
            ani.event_source.start()
            btn_play.label.set_text('|| Pause')
        is_playing[0] = not is_playing[0]
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    ani = animation.FuncAnimation(
        fig, _animation_update, frames=num_steps, interval=180, blit=False)

    # Initial draw
    update(0)

    if mode == "save":
        print(f"{prefix}Commencing 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Save complete.")
        except Exception as e:
            print(f"{prefix}Failed to save. ffmpeg installed? Error: {e}")
    else:
        print(f"{prefix}Opening GUI...  SPACEBAR = PNG snapshot.")
        plt.show()


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    mp.set_start_method('spawn', force=True)
    print("Forking 4K render to background process...")
    render_process = mp.Process(target=run_dashboard, args=("save",))
    render_process.start()

    run_dashboard(mode="interactive")

    if render_process.is_alive():
        print("\nInteractive viewer closed. "
              "Waiting for background 4K render to finish...")
        render_process.join()
    print("All processes terminated.")


if __name__ == "__main__":
    main()
