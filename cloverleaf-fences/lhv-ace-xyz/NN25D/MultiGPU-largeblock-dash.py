# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_v9_blocks.py
# Dashboard adapted for the Rev 88-C BLOCK LATTICE engine variant
# (16-qubit 4x4 brane tiles -> 4x4x4 brane-stack blocks -> 4x4x4 block
# lattice: 256 patches, 4096 qubits total).
#
# CHANGES vs v7 (brane-stack dashboard):
#   - GEOMETRY: the single 1x1xN brane stack is replaced by the full
#     block lattice. Patch addressing mirrors the engine exactly:
#     p = (tx * grid_y + ty) * grid_z + z, with (tx, ty) the in-plane
#     tile/block coordinates and z the global layer index. Within a
#     tile, idx = x * 4 + y (matching generate_16q_brane_tile()).
#     Tiles are separated laterally by TILE_GAP_XY so the coupled XY
#     seams are visible, and an extra BLOCK_GAP_Z is inserted between
#     blocks along Z so the block/"hadron" grouping reads at a glance.
#   - CONFIG DRIVEN: block_grid, branes_per_block, grid_x/y/z,
#     qubits_per_patch, and the periodic [X, Y, Z] triple are read from
#     lattice_config.json (with Rev 88-C defaults as fallbacks).
#     tile_geometry is checked against "4x4_brane_block_lattice".
#   - INTERFACES: rebuilt as a faithful mirror of the engine's
#     build_interfaces(): all three coupling classes (Z_INTRA, Z_INTER,
#     XY) with their exact per-interface site-index arrays. Z seams pair
#     (i, i) over all 16 sites; XY seams pair the 4-site +X/-X or +Y/-Y
#     tile edges. Periodic wraps (with the size-2 double-count guard)
#     are included in the metric, mirroring the engine.
#   - SITE-RESOLVED COUPLING ERROR: generalized from v7 to arbitrary
#     paired index arrays:
#         D(t, iface) = mean_k ||s_{i1[k]}^(p1)(t) - s_{i2[k]}^(p2)(t)||_2
#     and reported PER COUPLING CLASS (Z_INTRA / Z_INTER / XY curves,
#     plus the overall mean). As in v7 this is an internal
#     self-consistency measure of the mean-field stitching, not a
#     residual against any external ground truth. NOTE: for XY seams the
#     paired sites are distinct physical lattice sites (edge columns of
#     neighboring tiles), so a nonzero baseline is expected wherever the
#     in-plane profile has structure; watch its trend, not its absolute
#     level, and compare it against the Z classes with that in mind.
#   - ENERGY PANEL: reads the Rev 88-C per-class columns (E_Z_Intra,
#     E_Z_Inter, E_XY) in addition to the unchanged MeanField_* totals,
#     and overlays the class components as thin dashed lines.
#   - INTERFACE PLANES: v7 drew one sheet per open interface; at 624
#     interfaces that is visual noise. Only the inter-block Z boundaries
#     (the "weak glue" seams) are drawn, as full-footprint translucent
#     sheets in each inter-block gap. Z_INTRA and XY seams are counted
#     in the metrics but not drawn.
#   - HEATMAPS: 256 rows x 16 columns per component. Per-row separators
#     are dropped (unreadable at 256 rows); instead a strong separator
#     is drawn between tile columns (every grid_z rows) and a faint one
#     between blocks within a tile column (every branes_per_block rows).
#     Y ticks label each tile column "T(tx,ty)" at its center row.
#   - QUIVER STRIDE: 4096 arrows re-drawn per frame is heavy for the
#     interactive Axes3D path. QUIVER_STRIDE subsamples the drawn
#     arrows (metric panels always use the full data); default 1 draws
#     everything, set 2 or 4 if interaction is sluggish.
#
# Unchanged vs v7:
#   - ENERGY_FILE name and MeanField_* column names.
#   - _safe_gradient NaN-padding handling, slider/play/scroll/snapshot
#     UI, dual interactive + background-4K-render process model,
#     blit=False (Axes3D does not support blitting).

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
DATA_FILE      = "macroscopic_lattice_states.npy"
CONFIG_FILE    = "lattice_config.json"
ENERGY_FILE    = "meanfield_ground_state_energy_curve_multi.csv"
SAVE_FILE      = "macroscopic_lattice_dash_blocks.mp4"

TILE_LX, TILE_LY = 4, 4        # in-plane brane tile extent (matches engine)
LAYER_SPACING    = 1.6         # visual Z gap between adjacent branes
BLOCK_GAP_Z      = 1.2         # EXTRA visual Z gap at inter-block seams
TILE_GAP_XY      = 1.5         # visual lateral gap between neighbor tiles
QUIVER_STRIDE    = 1           # draw every Nth arrow (1 = all 4096)

KIND_COLORS = {"Z_INTRA": "#e6b422", "Z_INTER": "#e64550", "XY": "#45b0e6"}
# ---------------------


def load_energy_data(num_steps, log_prefix=""):
    """Extracts energy components (incl. Rev 88-C per-class columns)."""
    energies = {'Total': [], 'Bulk': [], 'Boundary': [],
                'Z_INTRA': [], 'Z_INTER': [], 'XY': []}
    kind_cols = {'Z_INTRA': "E_Z_Intra", 'Z_INTER': "E_Z_Inter", 'XY': "E_XY"}

    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies['Total'].append(float(row["MeanField_Total_Energy"]))
                energies['Bulk'].append(float(row["MeanField_Bulk_Energy"]))
                energies['Boundary'].append(float(row["MeanField_Boundary_Energy"]))
                for kind, col in kind_cols.items():
                    # Per-class columns are new in Rev 88-C; degrade to NaN
                    # gracefully if pointed at an older CSV.
                    try:
                        energies[kind].append(float(row[col]))
                    except (KeyError, ValueError):
                        energies[kind].append(np.nan)
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
    Compute np.gradient only over valid (non-NaN) indices.
    Returns a full-length array with NaN in positions where input was NaN.
    Handles mid-sequence NaNs safely.
    """
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 1:
        valid_data = arr[valid_indices]
        # Pass valid_indices to account for true step distances if mid-sequence gaps exist
        grad = np.gradient(valid_data, valid_indices)
        out[valid_indices] = grad
    elif len(valid_indices) == 1:
        out[valid_indices[0]] = 0.0
        
    return out


def build_interfaces(grid_x, grid_y, grid_z, branes_per_block, qpp,
                     periodic_x, periodic_y, periodic_z):
    """Faithful mirror of the Rev 88-C engine's build_interfaces().

    Returns a list of (p1, p2, idx1, idx2, kind). idx1[k] on p1 pairs
    with idx2[k] on p2. Kinds: Z_INTRA, Z_INTER, XY.
    """
    def patch_id(tx, ty, z):
        return (tx * grid_y + ty) * grid_z + z

    z_i1 = np.arange(qpp)                          # (i, i), 16 sites
    z_i2 = np.arange(qpp)                          # independent array to prevent mutation bleed
    x_i1 = np.array([12 + y for y in range(4)])    # +X face of p1
    x_i2 = np.array([y for y in range(4)])         # -X face of p2
    y_i1 = np.array([x * 4 + 3 for x in range(4)]) # +Y face of p1
    y_i2 = np.array([x * 4 for x in range(4)])     # -Y face of p2

    interfaces = []
    for tx in range(grid_x):
        for ty in range(grid_y):
            for z in range(grid_z):
                p1 = patch_id(tx, ty, z)

                # --- Z neighbor (brane stacking) ---
                if z < grid_z - 1:
                    kind = "Z_INTRA" if (z + 1) % branes_per_block != 0 else "Z_INTER"
                    interfaces.append((p1, patch_id(tx, ty, z + 1), z_i1, z_i2, kind))
                elif periodic_z and grid_z > 2:
                    # Wrap seam is a block boundary by construction
                    interfaces.append((p1, patch_id(tx, ty, 0), z_i1, z_i2, "Z_INTER"))

                # --- X neighbor (lateral block seam) ---
                if tx < grid_x - 1:
                    interfaces.append((p1, patch_id(tx + 1, ty, z), x_i1, x_i2, "XY"))
                elif periodic_x and grid_x > 2:
                    interfaces.append((p1, patch_id(0, ty, z), x_i1, x_i2, "XY"))

                # --- Y neighbor (lateral block seam) ---
                if ty < grid_y - 1:
                    interfaces.append((p1, patch_id(tx, ty + 1, z), y_i1, y_i2, "XY"))
                elif periodic_y and grid_y > 2:
                    interfaces.append((p1, patch_id(tx, 0, z), y_i1, y_i2, "XY"))

    return interfaces


def run_dashboard(mode="interactive"):
    """
    Main visualization routine.
    mode can be "interactive" (opens UI) or "save" (renders to disk headless).
    """
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "

    # 1. Dynamically load lattice configuration
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
    
    # Safely handle the periodic flag (ensure it translates correctly if passed as a single boolean or integer)
    periodic         = config.get("periodic", [False, False, False])
    if not isinstance(periodic, (list, tuple)):
        periodic = [bool(periodic)] * 3
    periodic_x, periodic_y, periodic_z = (bool(v) for v in periodic)

    if tile_geometry != "4x4_brane_block_lattice":
        print(f"{prefix}Warning: tile_geometry is '{tile_geometry}', "
              f"expected '4x4_brane_block_lattice'. Layout may be wrong.")

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
    expected_patches = grid_x * grid_y * grid_z
    if num_patches != expected_patches:
        print(f"{prefix}Warning: config implies {expected_patches} patches "
              f"({grid_x}x{grid_y}x{grid_z}) but state dump has {num_patches}; "
              f"trusting the state dump. Patch->coordinate mapping may be wrong.")
    total_qubits = num_patches * qpp

    def patch_coords(p):
        """Inverse of the engine's patch_id: (tx, ty, z)."""
        z = p % grid_z
        rest = p // grid_z
        return rest // grid_y, rest % grid_y, z

    # 3. Load energies, build interfaces, compute site-resolved coupling error
    energies = load_energy_data(num_steps, prefix)

    interfaces = build_interfaces(grid_x, grid_y, grid_z, branes_per_block,
                                  qpp, periodic_x, periodic_y, periodic_z)
    kinds_present = ["Z_INTRA", "Z_INTER", "XY"]
    n_by_kind = {k: sum(1 for itf in interfaces if itf[4] == k) for k in kinds_present}
    print(f"{prefix}Interfaces: {n_by_kind['Z_INTRA']} Z-intra, "
          f"{n_by_kind['Z_INTER']} Z-inter, {n_by_kind['XY']} XY "
          f"({len(interfaces)} total)")

    # Site-resolved coupling error, straight from the state dump, using the
    # exact paired site-index arrays of each interface:
    #   D[t, iface] = mean_k || s_{i1[k]}^(p1)(t) - s_{i2[k]}^(p2)(t) ||_2
    # NOTE: this is NOT a residual against any external reference. It measures
    # internal self-consistency of the site-resolved mean-field stitching
    # across each seam, NOT simulation accuracy relative to a ground truth.
    # XY seams pair DISTINCT physical sites (neighboring tile edges), so a
    # structured in-plane profile gives them a legitimate nonzero baseline.
    n_ifaces = max(len(interfaces), 1)
    disagreements = np.zeros((num_steps, n_ifaces))
    iface_kind = []
    if interfaces:
        hist_arr = np.asarray(history)  # mmap-safe view for vectorised math
        # Future optimization note: This loops serially over all interfaces. For very large
        # grids, consider batching the Z and XY slice indexing for faster startup.
        for i, (p1, p2, i1, i2, kind) in enumerate(interfaces):
            diff = hist_arr[:, p1, i1, :] - hist_arr[:, p2, i2, :]  # (steps, n_pairs, 3)
            disagreements[:, i] = np.mean(np.linalg.norm(diff, axis=2), axis=1)
            iface_kind.append(kind)
    iface_kind = np.array(iface_kind) if iface_kind else np.array([], dtype=str)

    avg_disagreement = (np.mean(disagreements, axis=1)
                        if interfaces else np.zeros(num_steps))
    dis_by_kind = {}
    for k in kinds_present:
        mask = (iface_kind == k)
        if np.any(mask):
            dis_by_kind[k] = np.mean(disagreements[:, mask], axis=1)

    # _safe_gradient: avoids NaN bleed from trailing padding into valid values.
    dE_dt   = _safe_gradient(energies['Total'])
    dRes_dt = _safe_gradient(avg_disagreement)

    # 4. Setup Global 3D Coordinates
    # Within a tile: idx = x * TILE_LY + y (row-major in x), matching
    # generate_16q_brane_tile(). Tile (tx, ty) is offset laterally by
    # TILE_GAP_XY; layer z sits at z * LAYER_SPACING plus an extra
    # BLOCK_GAP_Z for every completed block below it.
    tile_pitch_x = TILE_LX - 1 + TILE_GAP_XY + 1   # center-to-center X pitch
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

    # Optional arrow subsampling for the interactive 3D panel only.
    stride = max(1, int(QUIVER_STRIDE))
    draw_idx = np.arange(0, num_patches * qpp, stride)
    qX, qY, qZ = global_X[draw_idx], global_Y[draw_idx], global_Z[draw_idx]

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
            history[step_idx, :, :, 0].ravel()[draw_idx],
            history[step_idx, :, :, 1].ravel()[draw_idx],
            history[step_idx, :, :, 2].ravel()[draw_idx],
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
        qX, qY, qZ, U, V, W,
        length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    # Faint translucent full-footprint sheets ONLY at the inter-block Z
    # gaps (the weak-glue Z_INTER seams), as a visual cue for the block
    # grouping. Z_INTRA and XY seams (and any periodic wraps) are counted
    # in the error metric but not drawn: 624 sheets would be visual noise.
    margin = 0.4
    for zb in range(1, grid_z // branes_per_block):
        z_lo = z_pos(zb * branes_per_block - 1)
        z_hi = z_pos(zb * branes_per_block)
        z_mid = 0.5 * (z_lo + z_hi)
        verts = [[
            (-margin,         -margin,         z_mid),
            (x_max + margin,  -margin,         z_mid),
            (x_max + margin,  y_max + margin,  z_mid),
            (-margin,          y_max + margin, z_mid),
        ]]
        plane = Poly3DCollection(verts, alpha=0.06, facecolor='#e64550',
                                 edgecolor='#e64550', linewidths=0.5)
        ax3d.add_collection3d(plane)

    # 6. Colorbar - driven by spin_norm so ticks span the true [-1, +1] range
    ax_cbar = fig.add_axes([0.02, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Bloch Vector Component (Spin State)', fontsize=10)

    energy_text = ax3d.text2D(
        0.04, 0.96, "", transform=ax3d.transAxes,
        color='lightgreen', fontsize=12, fontweight='bold'
    )

    bx, by, bz = (block_grid + [4, 4, 4])[:3] if isinstance(block_grid, list) else (4, 4, 4)
    per_flags = ''.join(a for a, on in zip('XYZ', (periodic_x, periodic_y, periodic_z)) if on)
    lattice_desc = (f"{bx}x{by}x{bz} blocks x {branes_per_block} branes"
                    + (f" | periodic {per_flags}" if per_flags else ""))
    ax3d.set_title(
        f"Block-Lattice Annealing ({lattice_desc} | {num_patches} Branes | "
        f"{total_qubits} Qubits)\nTrotter Step: 0/{num_steps-1}",
        fontsize=14, pad=10
    )
    ax3d.set_xlim(-0.5, x_max + 0.5)
    ax3d.set_ylim(-0.5, y_max + 0.5)
    ax3d.set_zlim(-0.5, z_max + 0.5)
    try:
        ax3d.set_box_aspect((x_max + 1.0, y_max + 1.0, max(1.0, z_max + 1.0)))
    except AttributeError:
        pass  # set_box_aspect requires matplotlib >= 3.3

    # Hide all pane surfaces, pane edges, tick lines, tick labels, and the 3D grid
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor('none')
        except AttributeError:
            # Forward-compatibility guard (e.g. if pane.fill is removed in future Matplotlib builds)
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis.line.set_linewidth(0)
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()

    # --- 2D Analytics ---
    ax_energy.plot(energies['Total'], label='Total Energy', color='lightgreen')
    ax_energy.plot(energies['Bulk'], label='Bulk', color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Boundary (all)', color='orange')
    # Rev 88-C per-class boundary components (NaN if reading an older CSV)
    for k, lbl in (("Z_INTRA", "E Z-intra"), ("Z_INTER", "E Z-inter"), ("XY", "E XY")):
        if not np.all(np.isnan(energies[k])):
            ax_energy.plot(energies[k], label=lbl, color=KIND_COLORS[k],
                           linestyle='--', linewidth=0.9, alpha=0.9)
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.set_ylabel("Energy (a.u.)", fontsize=8)
    ax_energy.legend(fontsize=6, loc='upper left', ncol=2)
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    vline_d = None

    if not interfaces:
        ax_dis.set_visible(False)
    else:
        ax_dis.plot(avg_disagreement, color='crimson', linewidth=1.6,
                    label='Mean (all seams)')
        for k, lbl in (("Z_INTRA", "Mean Z-intra"), ("Z_INTER", "Mean Z-inter"), ("XY", "Mean XY")):
            if k in dis_by_kind:
                ax_dis.plot(dis_by_kind[k], color=KIND_COLORS[k],
                            linewidth=0.9, alpha=0.9, label=lbl)
        ax_dis.set_title("Coupling Error by Seam Class (site-resolved)\n"
                         "mean_seams( mean_k ||s_{i1[k]} - s_{i2[k]}|| )",
                         fontsize=9)
        ax_dis.set_ylabel("Error (L2 Norm)", fontsize=8)
        ax_dis.legend(fontsize=6, loc='upper left', ncol=2)
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    l1 = ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("dE/dt (per step)", fontsize=8)

    ax_deriv_r = ax_deriv.twinx()
    l2 = ax_deriv_r.plot(dRes_dt, label='d||ds||/dt', color='crimson')
    ax_deriv_r.set_ylabel("d||ds||/dt (per step)", fontsize=8)

    # Combine legends to prevent them overlapping if data spikes wildly
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_deriv.legend(lines, labels, loc='upper left', fontsize=8)

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

    # 256 rows: patch order is (tx, ty)-major, z-minor, so each contiguous
    # run of grid_z rows is one tile column across all layers. Label each
    # tile column at its center row; separate tile columns with a strong
    # line and blocks within a column with a faint one.
    y_ticks, y_labels = [], []
    for tx in range(grid_x):
        for ty in range(grid_y):
            base = (tx * grid_y + ty) * grid_z
            y_ticks.append(base + (grid_z - 1) / 2.0)
            y_labels.append(f"T({tx},{ty})")

    def _init_heatmap(ax, data, cmap, norm, label, show_ylabel=False, show_xlabel=False):
        """Create an imshow panel for the given data slice."""
        img = ax.imshow(
            data, cmap=cmap, norm=norm,
            aspect='auto', interpolation='nearest'
        )
        for r in range(branes_per_block, num_patches, branes_per_block):
            if r % grid_z == 0:
                continue  # tile-column boundary drawn below, stronger
            ax.axhline(r - 0.5, color='white', linewidth=0.3, alpha=0.35)
        for r in range(grid_z, num_patches, grid_z):
            ax.axhline(r - 0.5, color='white', linewidth=0.9, alpha=0.9)
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=4)
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
                current_aspect = (x_max + 1.0, y_max + 1.0, max(1.0, z_max + 1.0))
            ax3d.set_box_aspect(current_aspect, zoom=ax3d.custom_zoom)
        except TypeError:
            # Fallback for Matplotlib < 3.8 where zoom kwarg doesn't exist
            # Scroll down (event.button == 'down') conventionally zooms out (distance increases)
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
                f"dash_snapshot_blocks_{bx}x{by}x{bz}x{branes_per_block}"
                f"_step{current_step}_E{e_str}.png"
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
            qX, qY, qZ, U, V, W,
            length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3
        )

        ax3d.set_title(
            f"Block-Lattice Annealing ({lattice_desc} | {num_patches} Branes | "
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
