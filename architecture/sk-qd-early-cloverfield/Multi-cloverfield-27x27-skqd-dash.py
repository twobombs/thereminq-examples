# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_v7.py
#
# Changes vs v6 -- aligned to Rev 89 engine outputs:
#
# BUGFIXES:
#   - MEASURE_EVERY INDEX BUG: profiles_csv logs Trotter step t, but history is
#     indexed by measure position (0, 1, 2, ...). When measure_every > 1 these
#     diverge. load_analytics_data now builds a step->frame index map from the
#     actual Step values in the energy CSV, and profiles are keyed by frame index,
#     not by raw Trotter step number.
#
#   - QUBITS_PER_PATCH HARDCODE: q_coords was hardcoded as 3x3x3 (27 qubits).
#     Now derived from config["qubits_per_patch"]. The local qubit layout is
#     reconstructed from cbrt(qubits_per_patch) for cubic patches.
#
# NEW FEATURES:
#   - SKQD PANEL: New right-column panel reads skqd_refined_energies.csv and
#     displays per-patch SKQD_E0 vs MeanField_Bulk_E as a horizontal bar chart,
#     with Delta_E (correlation energy recovered) annotated. Only rendered if
#     the file exists and is non-empty.
#
#   - ANNEAL SCHEDULE OVERLAY: Anneal_Percent column is now read from the energy
#     CSV and overlaid as a dashed grey line on the energy panel (right axis).
#
#   - SKQD HEATMAP OVERLAY: When SKQD data is present, the 3D quiver view
#     optionally tints patch centroids by Delta_E magnitude (toggled via 'S' key).
#
# RETAINED from v6:
#   - All filename constants match Rev 89 engine (energy_csv, profiles_csv,
#     skqd_csv, state_dump_file, config_file).
#   - All CSV column names match Rev 89 _log_csvs() / _init_files() fieldnames.
#   - _safe_gradient, scroll zoom, spacebar screenshot, play/pause.
#   - Background 4K FFmpeg render in parallel with interactive viewer.

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
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
DATA_FILE     = "macroscopic_lattice_states.npy"
CONFIG_FILE   = "lattice_config.json"
ENERGY_FILE   = "meanfield_ground_state_energy_curve_multi.csv"
PROFILES_FILE = "boundary_profiles_multi.csv"
SKQD_FILE     = "skqd_refined_energies.csv"
SAVE_FILE     = "macroscopic_lattice_dash.mp4"
# ---------------------


def load_analytics_data(num_frames, log_prefix=""):
    """
    Load energy components, anneal schedule, and boundary profiles.

    num_frames: number of frames in the history array (history.shape[0]).

    The energy CSV has one row per measured Trotter step. Its "Step" column
    holds the raw Trotter index (0, measure_every, 2*measure_every, ...).
    We build a step->frame_index map so that profiles (also keyed by raw
    Trotter step) can be re-keyed to frame indices, fixing the measure_every>1
    index mismatch that existed in v6.
    """
    energies = {
        'Total':    [],
        'Bulk':     [],
        'Boundary': [],
        'Anneal':   [],   # Anneal_Percent column, new in v7
    }
    step_to_frame = {}   # raw Trotter step -> frame index in history array

    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for frame_idx, row in enumerate(reader):
                energies['Total'].append(float(row["MeanField_Total_Energy"]))
                energies['Bulk'].append(float(row["MeanField_Bulk_Energy"]))
                energies['Boundary'].append(float(row["MeanField_Boundary_Energy"]))
                energies['Anneal'].append(float(row.get("Anneal_Percent", float('nan'))))
                trotter_step = int(row["Step"])
                step_to_frame[trotter_step] = frame_idx
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE}: {e}")

    # Pad all arrays to num_frames with NaN
    for key in energies:
        deficit = num_frames - len(energies[key])
        if deficit > 0:
            energies[key].extend([float('nan')] * deficit)
        energies[key] = energies[key][:num_frames]

    # Load profiles, re-keyed to frame index (not raw Trotter step)
    profiles = {}
    try:
        with open(PROFILES_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trotter_step = int(row["Step"])
                frame_idx = step_to_frame.get(trotter_step)
                if frame_idx is None:
                    # step_to_frame is empty (energy CSV missing): fall back to
                    # treating Step as a frame index directly (measure_every=1 case)
                    frame_idx = trotter_step
                p = int(row["Patch"])
                f_name = row["Face"]
                if frame_idx not in profiles:
                    profiles[frame_idx] = {}
                if p not in profiles[frame_idx]:
                    profiles[frame_idx][p] = {}
                profiles[frame_idx][p][f_name] = np.array([
                    float(row["X_mean"]), float(row["Y_mean"]), float(row["Z_mean"])
                ])
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {PROFILES_FILE}: {e}")

    return energies, profiles


def load_skqd_data(num_patches, log_prefix=""):
    """
    Load per-patch SKQD refinement results from skqd_refined_energies.csv.

    Returns arrays indexed 0..num_patches-1. Patches with no SKQD entry
    (e.g. no samples captured) are filled with NaN.

    Columns consumed:
        Patch, Seed_States, Final_Subspace, SKQD_E0, MeanField_Bulk_E, Delta_E
    """
    skqd_e0   = np.full(num_patches, float('nan'))
    mf_bulk_e = np.full(num_patches, float('nan'))
    delta_e   = np.full(num_patches, float('nan'))
    seed_n    = np.zeros(num_patches, dtype=int)
    sub_n     = np.zeros(num_patches, dtype=int)
    found_any = False

    try:
        with open(SKQD_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = int(row["Patch"])
                if 0 <= p < num_patches:
                    skqd_e0[p]   = float(row["SKQD_E0"])
                    mf_bulk_e[p] = float(row["MeanField_Bulk_E"])
                    delta_e[p]   = float(row["Delta_E"])
                    seed_n[p]    = int(row["Seed_States"])
                    sub_n[p]     = int(row["Final_Subspace"])
                    found_any    = True
    except FileNotFoundError:
        pass   # SKQD output is optional; panel will be hidden
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {SKQD_FILE}: {e}")

    return (skqd_e0, mf_bulk_e, delta_e, seed_n, sub_n) if found_any else None


def _safe_gradient(arr_with_nans):
    """
    Compute np.gradient only over the leading valid (non-NaN) slice.
    Returns a full-length array with NaN in positions where input was NaN.
    """
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, float('nan'))
    valid_mask = ~np.isnan(arr)
    valid_count = int(np.sum(valid_mask))
    if valid_count > 1:
        out[:valid_count] = np.gradient(arr[:valid_count])
    elif valid_count == 1:
        out[:1] = 0.0
    return out


def _build_q_coords(qubits_per_patch):
    """
    Build local qubit -> (x, y, z) coordinate map for a cubic patch.
    Assumes qubits_per_patch is a perfect cube. Falls back to 1D layout
    if cbrt is not integer.
    """
    side = round(qubits_per_patch ** (1.0 / 3.0))
    if side ** 3 != qubits_per_patch:
        # Non-cubic patch: lay qubits out along X axis
        return {q: (q, 0, 0) for q in range(qubits_per_patch)}
    coords = {}
    for x in range(side):
        for y in range(side):
            for z in range(side):
                coords[x * side * side + y * side + z] = (x, y, z)
    return coords


def run_dashboard(mode="interactive"):
    """
    Main visualization routine.
    mode: "interactive" (opens UI) or "save" (renders to disk headless).
    """
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "

    # --- 1. Load config ---
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    grid_x          = config.get("grid_x", 3)
    grid_y          = config.get("grid_y", 3)
    grid_z          = config.get("grid_z", 3)
    num_patches      = config.get("num_patches", grid_x * grid_y * grid_z)
    qubits_per_patch = config.get("qubits_per_patch", 27)

    # --- 2. Load state history ---
    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    # Validate shape against config
    num_frames = history.shape[0]
    if history.shape[1] != num_patches:
        print(f"{prefix}Warning: history has {history.shape[1]} patches but config says {num_patches}. "
              f"Using history shape.")
        num_patches = history.shape[1]
    if history.shape[2] != qubits_per_patch:
        print(f"{prefix}Warning: history has {history.shape[2]} qubits/patch but config says "
              f"{qubits_per_patch}. Using history shape.")
        qubits_per_patch = history.shape[2]

    total_qubits = num_patches * qubits_per_patch

    # --- 3. Load analytics data ---
    energies, profiles = load_analytics_data(num_frames, prefix)
    skqd_data = load_skqd_data(num_patches, prefix)
    has_skqd = skqd_data is not None

    # --- 4. Patch / qubit coordinate setup ---
    patch_coords = {}
    coord_to_patch = {}
    idx = 0
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                patch_coords[idx] = (x, y, z)
                coord_to_patch[(x, y, z)] = idx
                idx += 1

    q_coords = _build_q_coords(qubits_per_patch)
    patch_side = round(qubits_per_patch ** (1.0 / 3.0))

    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        px, py, pz = patch_coords[p]
        for q in range(qubits_per_patch):
            qx, qy, qz = q_coords[q]
            global_X.append(px * patch_side + qx)
            global_Y.append(py * patch_side + qy)
            global_Z.append(pz * patch_side + qz)
    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    # --- 5. Interface / disagreement calculation ---
    interfaces = []
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                p1 = coord_to_patch[(x, y, z)]
                if x < grid_x - 1:
                    p2 = coord_to_patch[(x+1, y, z)]
                    interfaces.append((p1, p2, "+X", "-X",
                                       x*patch_side+patch_side-0.5, y*patch_side+patch_side/2, z*patch_side+patch_side/2))
                if y < grid_y - 1:
                    p2 = coord_to_patch[(x, y+1, z)]
                    interfaces.append((p1, p2, "+Y", "-Y",
                                       x*patch_side+patch_side/2, y*patch_side+patch_side-0.5, z*patch_side+patch_side/2))
                if z < grid_z - 1:
                    p2 = coord_to_patch[(x, y, z+1)]
                    interfaces.append((p1, p2, "+Z", "-Z",
                                       x*patch_side+patch_side/2, y*patch_side+patch_side/2, z*patch_side+patch_side-0.5))

    disagreements = np.zeros((num_frames, max(len(interfaces), 1)))
    for frame_idx in range(num_frames):
        if frame_idx in profiles:
            for i, (p1, p2, f1, f2, _, _, _) in enumerate(interfaces):
                try:
                    v1 = profiles[frame_idx][p1][f1]
                    v2 = profiles[frame_idx][p2][f2]
                    disagreements[frame_idx, i] = np.linalg.norm(v1 - v2)
                except KeyError:
                    pass

    avg_disagreement = np.mean(disagreements, axis=1) if interfaces else np.zeros(num_frames)

    dE_dt   = _safe_gradient(energies['Total'])
    dRes_dt = _safe_gradient(avg_disagreement)

    # --- 6. Figure layout ---
    # Right column rows:
    #   0: Energy + anneal overlay
    #   1: Inter-patch coupling error
    #   2: Derivatives
    #   3: Heatmap X
    #   4: Heatmap Y
    #   5: Heatmap Z
    #   6: SKQD panel (only if SKQD data present)
    n_right_rows = 7 if has_skqd else 6

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10 + (1.5 if has_skqd else 0)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)
    ax3d = fig.add_subplot(gs[0], projection='3d')
    gs_right = gridspec.GridSpecFromSubplotSpec(
        n_right_rows, 1, subplot_spec=gs[1],
        hspace=0.85, height_ratios=([1]*6 + [1.6]) if has_skqd else [1]*6
    )
    ax_energy  = fig.add_subplot(gs_right[0])
    ax_dis     = fig.add_subplot(gs_right[1])
    ax_deriv   = fig.add_subplot(gs_right[2])
    ax_hmap_x  = fig.add_subplot(gs_right[3])
    ax_hmap_y  = fig.add_subplot(gs_right[4])
    ax_hmap_z  = fig.add_subplot(gs_right[5])
    ax_skqd    = fig.add_subplot(gs_right[6]) if has_skqd else None

    # --- 7. Colormaps ---
    spin_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    vector_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ghost_vectors",
        [(0.15, 0.35, 0.85, 0.85), (0.85, 0.85, 0.85, 0.45), (0.85, 0.15, 0.25, 0.85)]
    )
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap_cmap",
        [(0.15, 0.35, 0.85, 1.0), (0.10, 0.10, 0.10, 1.0), (0.85, 0.15, 0.25, 1.0)]
    )

    # Delta_E colormap for SKQD 3D tinting: white=0 (no correction), red=negative (energy recovered)
    delta_norm = None
    skqd_tint_colors = None
    if has_skqd:
        skqd_e0, mf_bulk_e, delta_e, seed_n, sub_n = skqd_data
        finite_deltas = delta_e[np.isfinite(delta_e)]
        if len(finite_deltas) > 0:
            d_abs_max = max(abs(finite_deltas.min()), abs(finite_deltas.max()), 1e-6)
            delta_norm = mcolors.Normalize(vmin=-d_abs_max, vmax=d_abs_max)

    def _quiver_colors(w_flat, use_skqd_tint=False):
        if use_skqd_tint and has_skqd and delta_norm is not None:
            # Tint each qubit by its patch's Delta_E
            patch_colors = plt.cm.RdBu(delta_norm(delta_e))  # shape (num_patches, 4)
            per_qubit = np.repeat(patch_colors, qubits_per_patch, axis=0)
            return per_qubit
        return vector_cmap(spin_norm(w_flat))

    def get_vector_data(step_idx):
        return (
            history[step_idx, :, :, 0].flatten(),
            history[step_idx, :, :, 1].flatten(),
            history[step_idx, :, :, 2].flatten(),
        )

    U, V, W = get_vector_data(0)
    quiver_obj = [ax3d.quiver(
        global_X, global_Y, global_Z, U, V, W,
        length=0.75, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    # SKQD tint toggle state
    skqd_tint_active = [False]

    # --- 8. Colorbar ---
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
        f"{num_patches} Patches | {total_qubits} Qubits)\nTrotter Step: 0/{num_frames-1}",
        fontsize=14, pad=10
    )
    ax3d.set_xlim(-0.5, grid_x * patch_side - 0.5)
    ax3d.set_ylim(-0.5, grid_y * patch_side - 0.5)
    ax3d.set_zlim(-0.5, grid_z * patch_side - 0.5)
    try:
        ax3d.set_box_aspect((grid_x, grid_y, max(1, grid_z)))
    except AttributeError:
        pass
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.line.set_linewidth(0)
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()

    # --- 9. Energy panel with anneal overlay ---
    ax_energy.plot(energies['Total'],    label='Total Energy', color='lightgreen')
    ax_energy.plot(energies['Bulk'],     label='Bulk',         color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Boundary',     color='orange')
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.legend(fontsize=7, loc='upper left')
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # Anneal schedule on twin axis (Anneal_Percent 0->100)
    anneal_arr = np.array(energies['Anneal'])
    if np.any(np.isfinite(anneal_arr)):
        ax_anneal = ax_energy.twinx()
        ax_anneal.plot(anneal_arr, color='#888888', linestyle=':', linewidth=1.0,
                       label='Anneal %')
        ax_anneal.set_ylim(-5, 105)
        ax_anneal.set_ylabel("Anneal %", fontsize=7, color='#888888')
        ax_anneal.tick_params(axis='y', labelsize=6, colors='#888888')
        ax_anneal.legend(fontsize=7, loc='lower right')

    # --- 10. Disagreement panel ---
    vline_d = None
    if interfaces:
        ax_dis.plot(avg_disagreement, color='crimson',
                    label='Mean ||d<s>|| across interfaces')
        ax_dis.set_title("Inter-patch Boundary Coupling Error\n"
                         "mean_ifaces( ||<s>+f - <s>-f|| )", fontsize=9)
        ax_dis.legend(fontsize=7, loc='upper left')
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # --- 11. Derivatives panel ---
    ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("Energy Delta", fontsize=8)
    ax_deriv.legend(loc='upper left', fontsize=7)
    ax_deriv_r = ax_deriv.twinx()
    ax_deriv_r.plot(dRes_dt, label='d||d<s>||/dt', color='crimson')
    ax_deriv_r.set_ylabel("Coupling Error dt", fontsize=8)
    ax_deriv_r.legend(loc='upper right', fontsize=7)
    ax_deriv.set_title("Derivatives (Convergence Rate)", fontsize=10)
    ax_deriv.grid(True, alpha=0.2)
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # --- 12. Heatmap panels ---
    y_ticks, y_labels = [], []
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
        img = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        for i in range(1, num_patches):
            if patch_coords[i][0] != patch_coords[i-1][0]:
                ax.axhline(i-0.5, color='white', linewidth=1.2, alpha=1.0)
            elif patch_coords[i][1] != patch_coords[i-1][1]:
                ax.axhline(i-0.5, color='#aaaaaa', linewidth=0.7, alpha=0.7, linestyle=':')
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=6)
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel("Local Qubit Index (0-%d)" % (qubits_per_patch-1), fontsize=8)
            ax.tick_params(axis='x', which='major', labelsize=7)
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0], heatmap_cmap, spin_norm,
                            "Polarization <X>", show_ylabel=True, show_xlabel=False)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1], heatmap_cmap, spin_norm,
                            "Polarization <Y>", show_ylabel=True, show_xlabel=False)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2], heatmap_cmap, spin_norm,
                            "Polarization <Z>", show_ylabel=True, show_xlabel=True)

    # --- 13. SKQD panel (static -- computed once at final step) ---
    if has_skqd and ax_skqd is not None:
        skqd_e0, mf_bulk_e, delta_e, seed_n, sub_n = skqd_data
        patch_labels = [
            f"P{p} ({patch_coords[p][0]},{patch_coords[p][1]},{patch_coords[p][2]})"
            for p in range(num_patches)
        ]
        y_pos = np.arange(num_patches)

        # SKQD_E0 and MF_Bulk side-by-side, finite values only styled
        finite_mask = np.isfinite(skqd_e0)
        ax_skqd.barh(y_pos[finite_mask], skqd_e0[finite_mask],
                     color='mediumpurple', alpha=0.8, height=0.4, label='SKQD E0',
                     align='center')
        ax_skqd.barh(y_pos[finite_mask] + 0.4, mf_bulk_e[finite_mask],
                     color='dodgerblue', alpha=0.6, height=0.4, label='MF Bulk E',
                     align='center')

        # Annotate Delta_E (correlation energy recovered) on right side
        ax_skqd_r = ax_skqd.twinx()
        ax_skqd_r.scatter(
            delta_e[finite_mask], y_pos[finite_mask] + 0.2,
            color='gold', s=18, zorder=5, label='Delta_E (correction)'
        )
        ax_skqd_r.axvline(x=0, color='#666666', linewidth=0.8, linestyle='--')
        ax_skqd_r.set_ylabel("Delta_E", fontsize=7, color='gold')
        ax_skqd_r.tick_params(axis='y', labelsize=6, colors='gold')
        ax_skqd_r.legend(fontsize=6, loc='lower right')

        # Subspace size annotation
        for p in range(num_patches):
            if finite_mask[p]:
                ax_skqd.text(
                    ax_skqd.get_xlim()[0] if ax_skqd.get_xlim()[0] != 0 else skqd_e0[finite_mask].min() * 1.02,
                    p,
                    f" sub={sub_n[p]}", fontsize=5, va='center', color='#aaaaaa'
                )

        ax_skqd.set_yticks(y_pos + 0.2)
        ax_skqd.set_yticklabels(patch_labels, fontsize=5)
        ax_skqd.set_title(
            "SKQD Refinement: E0 vs MF Bulk\n(Delta_E = correlation energy recovered)",
            fontsize=8
        )
        ax_skqd.legend(fontsize=6, loc='upper right')
        ax_skqd.grid(True, alpha=0.15, axis='x')

        skqd_tint_label = ax3d.text2D(
            0.04, 0.88, "", transform=ax3d.transAxes,
            color='gold', fontsize=9
        )
    else:
        skqd_tint_label = ax3d.text2D(
            0.04, 0.88, "", transform=ax3d.transAxes, color='gold', fontsize=9
        )

    # --- 14. Slider and play/pause ---
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    ax_slider = fig.add_axes([0.15, 0.04, 0.60, 0.02])
    slider = Slider(
        ax=ax_slider, label='Frame (Trotter Step)',
        valmin=0, valmax=num_frames - 1, valinit=0, valstep=1, color='#4a90e2'
    )

    ax_play = fig.add_axes([0.80, 0.025, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')
    is_playing = True

    # --- 15. Scroll zoom ---
    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        if not hasattr(ax3d, 'custom_zoom'):
            ax3d.custom_zoom = 1.0
        ax3d.custom_zoom *= 0.9 if event.button == 'down' else 1.1
        try:
            current_aspect = ax3d.get_box_aspect()
            if current_aspect is None:
                current_aspect = (grid_x, grid_y, max(1, grid_z))
            ax3d.set_box_aspect(current_aspect, zoom=ax3d.custom_zoom)
        except TypeError:
            ax3d.dist *= 0.9 if event.button == 'down' else 1.1
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # --- 16. Key bindings ---
    def on_key_press(event):
        if event.key == ' ':
            current_step = int(slider.val)
            e_list = energies['Total']
            e_val = e_list[current_step] if current_step < len(e_list) else float('nan')
            e_str = f"{e_val:.4f}" if (isinstance(e_val, float) and not np.isnan(e_val)) else "NaN"
            filename = (
                f"dash_snapshot_{grid_x}x{grid_y}x{grid_z}"
                f"_frame{current_step}_E{e_str}.png"
            )
            print(f"{prefix}Saving screenshot to {filename}...")
            fig.savefig(filename, dpi=600, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"{prefix}Screenshot saved.")
        elif event.key == 's' and has_skqd:
            # Toggle SKQD Delta_E tint on 3D quiver
            skqd_tint_active[0] = not skqd_tint_active[0]
            label = "SKQD Delta_E tint: ON (press S to toggle)" if skqd_tint_active[0] else ""
            skqd_tint_label.set_text(label)
            update(int(slider.val))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # --- 17. Update function ---
    _from_animation = [False]

    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)

        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            global_X, global_Y, global_Z, U, V, W,
            length=0.75,
            colors=_quiver_colors(W, use_skqd_tint=skqd_tint_active[0]),
            arrow_length_ratio=0.3
        )

        # Resolve raw Trotter step for title (best-effort from anneal array)
        anneal_val = energies['Anneal'][frame] if frame < len(energies['Anneal']) else float('nan')
        anneal_str = f" | Anneal: {anneal_val:.1f}%" if not math.isnan(anneal_val) else ""
        ax3d.set_title(
            f"Macroscopic Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
            f"{num_patches} Patches | {total_qubits} Qubits)\n"
            f"Frame: {frame}/{num_frames-1}{anneal_str}",
            fontsize=14, pad=10
        )

        e_list = energies['Total']
        if e_list and frame < len(e_list):
            e_val = e_list[frame]
            energy_text.set_text(
                f"Total Energy: {e_val:.4f}" if not math.isnan(e_val) else ""
            )
        else:
            energy_text.set_text("")

        vline_e.set_xdata([frame, frame])
        if vline_d is not None:
            vline_d.set_xdata([frame, frame])
        vline_deriv.set_xdata([frame, frame])

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

    ani = animation.FuncAnimation(
        fig, _animation_update, frames=num_frames, interval=150, blit=False
    )

    if mode == "save":
        print(f"{prefix}Commencing 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Save complete.")
        except Exception as e:
            print(f"{prefix}Failed to save. Is ffmpeg installed? Error: {e}")
    else:
        if has_skqd:
            print(f"{prefix}SKQD data loaded. Press 'S' to toggle Delta_E tint on 3D view.")
        print(f"{prefix}Opening GUI...")
        plt.show()


def main():
    mp.set_start_method('spawn', force=True)
    print("Forking 4K render to background process...")
    render_process = mp.Process(target=run_dashboard, args=("save",))
    render_process.start()
    run_dashboard(mode="interactive")
    if render_process.is_alive():
        print("\nInteractive viewer closed. Waiting for background 4K render...")
        render_process.join()
    print("All processes terminated.")


if __name__ == "__main__":
    main()
