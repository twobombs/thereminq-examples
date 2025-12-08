import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import re
from datetime import datetime

def show_help(message="No input file pattern provided."):
    """Prints a helpful CLI syntax message."""
    print(f"\nError: {message}")
    print("-" * 60)
    print("Usage: python anim_protein_folding.py <path_to_log_file_pattern>")
    print("-" * 60)
    print("\nThis script visualizes protein folding conformations from log files.")
    print("It generates an interactive 3D plot where:")
    print("  - The visualization is unboxed and floats in the center of the frame.")
    print("  - High-energy folds are thin/transparent, low-energy folds are thick/opaque.")
    print("  - You can drag with the left mouse button to rotate.")
    print("  - You can use the mouse wheel to zoom in and out.")
    print("  - You can right-click to save a high-resolution screenshot.")
    print("\nExample:")
    print("  python anim_protein_folding.py 'protein_folding_Insulin_*.log'")

def parse_coords(coord_string):
    """Parses the coordinate string into a list of (x, y) tuples."""
    try:
        pairs = re.findall(r'\((-?\d+\.?\d*),(-?\d+\.?\d*)\)', str(coord_string))
        return [(float(x), float(y)) for x, y in pairs]
    except (TypeError, ValueError):
        return []

def plot_folding_evolution(file_list):
    """
    Reads data from log files, sorts it, and creates a feature-rich, interactive 3D plot.
    """
    # --- CHANGE HERE: Decreased spacing factor for a more compact view ---
    spacing_factor = 0.1

    parsed_data = []
    for log_file in file_list:
        try:
            with open(log_file, 'r') as f:
                header = next(f).strip().split(',')
                for line in f:
                    try:
                        parts = line.strip().split(',', 2)
                        if len(parts) == 3:
                            parsed_data.append(parts)
                    except Exception:
                        continue
        except Exception as e:
            print(f"\nWarning: Could not process file '{log_file}'. Error: {e}")
            continue

    if not parsed_data:
        print("\nError: No valid data rows could be parsed from the specified files.")
        return

    combined_df = pd.DataFrame(parsed_data, columns=header)

    for col in ['run', 'energy']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    combined_df.dropna(subset=['run', 'energy', 'conformation_coords'], inplace=True)

    combined_df['coords'] = combined_df['conformation_coords'].apply(parse_coords)
    combined_df = combined_df[combined_df['coords'].apply(lambda c: isinstance(c, list) and len(c) > 1)]

    if combined_df.empty:
        print("\nError: No valid conformations remained after cleaning and parsing.")
        return

    df_sorted = combined_df.sort_values(by='energy', ascending=False).reset_index(drop=True)

    plt.style.use('dark_background')

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    norm = plt.Normalize(df_sorted['energy'].min(), df_sorted['energy'].max())
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)

    num_conformations = len(df_sorted)
    for index, row in df_sorted.iterrows():
        x_coords, y_coords = zip(*row['coords'])
        z_coords = index * spacing_factor
        line_color = mapper.to_rgba(row['energy'])
        
        current_alpha = 0.01 + (0.99 * (index / (num_conformations - 1)))
        current_linewidth = 0.5 + (1.0 * (index / (num_conformations - 1)))
        
        ax.plot(x_coords, y_coords, z_coords, color=line_color, linewidth=current_linewidth, alpha=current_alpha)

    # Add padding around the data to remove the 'boxed-in' effect
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    z_center = (zlim[0] + zlim[1]) / 2
    
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    plot_radius = 1.5 * max(x_range, y_range, z_range) / 2
    ax.set_xlim([x_center - plot_radius, x_center + plot_radius])
    ax.set_ylim([y_center - plot_radius, y_center + plot_radius])
    ax.set_zlim([z_center - plot_radius, z_center + plot_radius])
    
    ax.view_init(elev=30, azim=-65)
    ax.set_axis_off()

    cax = fig.add_axes([0.9, 0.6, 0.02, 0.3])
    cbar = fig.colorbar(mapper, cax=cax)
    cbar.ax.invert_yaxis()
    cbar.set_label('Energy Level', rotation=270, labelpad=15)

    def on_scroll(event):
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        for axis in ['x', 'y', 'z']:
            lim = getattr(ax, f'get_{axis}lim')()
            center = (lim[0] + lim[1]) / 2
            new_lower = center + (lim[0] - center) * scale_factor
            new_upper = center + (lim[1] - center) * scale_factor
            getattr(ax, f'set_{axis}lim')([new_lower, new_upper])

        fig.canvas.draw_idle()

    def on_click(event):
        if event.button == 3:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'folding_screenshot_{timestamp}.png'
            try:
                fig.savefig(filename, dpi=300, facecolor='black')
                print(f"\nSaved high-resolution screenshot as {filename}")
            except Exception as e:
                print(f"\nCould not save screenshot. Error: {e}")

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_click)

    print(f"Displaying interactive 3D plot from {len(file_list)} file(s).")
    print(f"Plotting {len(df_sorted)} valid conformations.")
    print("Drag to rotate, scroll to zoom, and right-click to save a high-res screenshot.")
    print("Close the plot window to exit the script.")
    
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)

    all_files = []
    for pattern in sys.argv[1:]:
        all_files.extend(glob.glob(pattern))

    unique_files = sorted(list(set(all_files)))

    if not unique_files:
        show_help(f"No files found matching the pattern(s) provided.")
        sys.exit(1)

    plot_folding_evolution(unique_files)
