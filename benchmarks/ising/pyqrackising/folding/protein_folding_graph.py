import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Set the plot style to dark mode ---
plt.style.use('dark_background')

def parse_filename(filename):
    """
    Extracts protein name, quality level, and device from a log file name.
    """
    pattern = r"protein_folding_(?P<protein>.*?)_(?P<mer>\d+mer)_q(?P<q_level>\d+)(?P<device>_cpu)?\.log"
    match = re.match(pattern, filename)
    
    if match:
        data = match.groupdict()
        device = 'cpu' if data['device'] else 'gpu'
        return {
            "protein": data['protein'],
            "q_level": int(data['q_level']),
            "device": device
        }
    return None

def read_first_conformation(filepath):
    """
    Reads a log file and returns the coordinate string from the first data row.
    """
    try:
        with open(filepath, 'r') as f:
            next(f)  # Skip the header line
            first_data_line = f.readline()
            parts = first_data_line.strip().split(',', 2)
            if len(parts) == 3:
                # Return the full line data: run, energy, coords
                return parts[1], parts[2]
    except Exception as e:
        print(f"  - [ERROR] Could not read file {os.path.basename(filepath)}: {e}")
    return None, None

def parse_coords(coord_string):
    """
    Converts the coordinate string from the log file into a list of (x, y) tuples.
    """
    coords = []
    if not isinstance(coord_string, str):
        return []

    parts = coord_string.strip().split(';')
    for part in parts:
        if not part:
            continue
        try:
            x, y = map(int, part.strip('()').split(','))
            coords.append((x, y))
        except ValueError:
            continue
    return coords

# --- Main Script Logic ---

log_files_dir = '.'
files_by_protein = defaultdict(lambda: defaultdict(dict))

for filename in sorted(os.listdir(log_files_dir)):
    if filename.startswith("protein_folding_") and filename.endswith(".log"):
        parsed_info = parse_filename(filename)
        if parsed_info:
            protein = parsed_info['protein']
            q_level = parsed_info['q_level']
            device = parsed_info['device']
            files_by_protein[protein][q_level][device] = os.path.join(log_files_dir, filename)

for protein, q_levels_data in files_by_protein.items():
    sorted_q_levels = sorted(q_levels_data.keys())
    num_rows = len(sorted_q_levels)
    
    if num_rows == 0:
        continue

    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows), squeeze=False)
    protein_title = protein.replace('_', ' ').title()
    fig.suptitle(f"'{protein_title}' Folded Conformations by Quality Level", fontsize=16, y=0.99)

    for i, q_level in enumerate(sorted_q_levels):
        # Process CPU file
        ax_cpu = axes[i, 0]
        cpu_file = q_levels_data[q_level].get('cpu')
        if cpu_file:
            energy, coord_string = read_first_conformation(cpu_file)
            if coord_string:
                coords = parse_coords(coord_string)
                if coords:
                    x_coords, y_coords = zip(*coords)
                    ax_cpu.plot(x_coords, y_coords, marker='o', linestyle='-', color='cyan')
                    # Add energy to the title
                    title = f"Quality: {q_level} (CPU)"
                    if energy:
                        title += f"\nEnergy: {float(energy):.2f}"
                    ax_cpu.set_title(title)
                    ax_cpu.set_aspect('equal', adjustable='box')
                else:
                    ax_cpu.set_title(f"Quality: {q_level} (CPU)\nNo coords parsed")
            else:
                ax_cpu.set_title(f"Quality: {q_level} (CPU)\nError reading coords")
        else:
            ax_cpu.set_title(f"Quality: {q_level} (CPU)\nNot Found")
            ax_cpu.axis('off')

        # Process GPU file
        ax_gpu = axes[i, 1]
        gpu_file = q_levels_data[q_level].get('gpu')
        if gpu_file:
            energy, coord_string = read_first_conformation(gpu_file)
            if coord_string:
                coords = parse_coords(coord_string)
                if coords:
                    x_coords, y_coords = zip(*coords)
                    ax_gpu.plot(x_coords, y_coords, marker='o', linestyle='-', color='yellow')
                    # Add energy to the title
                    title = f"Quality: {q_level} (GPU)"
                    if energy:
                         title += f"\nEnergy: {float(energy):.2f}"
                    ax_gpu.set_title(title)
                    ax_gpu.set_aspect('equal', adjustable='box')
                else:
                    ax_gpu.set_title(f"Quality: {q_level} (GPU)\nNo coords parsed")
            else:
                ax_gpu.set_title(f"Quality: {q_level} (GPU)\nError reading coords")
        else:
            ax_gpu.set_title(f"Quality: {q_level} (GPU)\nNot Found")
            ax_gpu.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # --- SAVE THE FIGURE ---
    # Create a filename like 'Glucagon_conformations.png'
    output_filename = f"{protein_title}_conformations.png"
    plt.savefig(output_filename, dpi=300) # dpi=300 gives high-quality output
    print(f"Plot saved to {output_filename}")
    
    plt.show()
