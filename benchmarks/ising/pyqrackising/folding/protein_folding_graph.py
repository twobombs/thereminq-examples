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
    pattern = r"protein_folding_(?P<protein>.*?)_(?P<mer>\d+mer)_q(?P<q_level>\d+)(?P<device>_cpu_on_gpu)?\.log"
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
    Reads a log file and returns the energy and coordinate string from the first data row.
    """
    try:
        with open(filepath, 'r') as f:
            next(f)  # Skip the header line
            first_data_line = f.readline()
            parts = first_data_line.strip().split(',', 2)
            if len(parts) == 3:
                # Return energy (as float) and coordinate string
                return float(parts[1]), parts[2]
    except (Exception, ValueError) as e:
        print(f"  - [ERROR] Could not read file {os.path.basename(filepath)}: {e}")
    return None, None

def parse_coords(coord_string):
    """
    Converts the coordinate string into a list of (x, y) tuples.
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
    # --- Step 1: Read data and determine the sorting order ---
    plot_rows_data = []
    for q_level, device_files in q_levels_data.items():
        cpu_file = device_files.get('cpu')
        gpu_file = device_files.get('gpu')
        
        cpu_energy, _ = read_first_conformation(cpu_file) if cpu_file else (None, None)
        gpu_energy, _ = read_first_conformation(gpu_file) if gpu_file else (None, None)

        # Determine the best energy for this quality level to use for sorting
        energies = [e for e in [cpu_energy, gpu_energy] if e is not None]
        if not energies:
            continue # Skip if no valid energy data was found
        
        min_energy = min(energies)
        
        plot_rows_data.append({
            'q_level': q_level,
            'min_energy': min_energy,
            'cpu_file': cpu_file,
            'gpu_file': gpu_file
        })

    # --- Step 2: Sort the rows by the minimum energy (lowest first) ---
    sorted_plot_rows = sorted(plot_rows_data, key=lambda x: x['min_energy'])

    num_rows = len(sorted_plot_rows)
    if num_rows == 0:
        continue

    # --- Step 3: Create the plot using the sorted data ---
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows), squeeze=False)
    protein_title = protein.replace('_', ' ').title()
    fig.suptitle(f"'{protein_title}' Folded Conformations (Sorted by Energy)", fontsize=16, y=0.99)

    for i, row_data in enumerate(sorted_plot_rows):
        q_level = row_data['q_level']
        
        # --- Column 0: CPU Plot ---
        ax_cpu = axes[i, 0]
        if row_data['cpu_file']:
            energy, coord_string = read_first_conformation(row_data['cpu_file'])
            if coord_string:
                coords = parse_coords(coord_string)
                if coords:
                    x_coords, y_coords = zip(*coords)
                    ax_cpu.plot(x_coords, y_coords, marker='o', linestyle='-', color='cyan')
                    title = f"Quality: {q_level} (CPU)\nEnergy: {energy:.2f}"
                    ax_cpu.set_title(title)
                    ax_cpu.set_aspect('equal', adjustable='box')
        else:
            ax_cpu.set_title(f"Quality: {q_level} (CPU)\nNot Found")
            ax_cpu.axis('off')

        # --- Column 1: GPU Plot ---
        ax_gpu = axes[i, 1]
        if row_data['gpu_file']:
            energy, coord_string = read_first_conformation(row_data['gpu_file'])
            if coord_string:
                coords = parse_coords(coord_string)
                if coords:
                    x_coords, y_coords = zip(*coords)
                    ax_gpu.plot(x_coords, y_coords, marker='o', linestyle='-', color='yellow')
                    title = f"Quality: {q_level} (GPU)\nEnergy: {energy:.2f}"
                    ax_gpu.set_title(title)
                    ax_gpu.set_aspect('equal', adjustable='box')
        else:
            ax_gpu.set_title(f"Quality: {q_level} (GPU)\nNot Found")
            ax_gpu.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_filename = f"{protein_title}_conformations_sorted_by_energy.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    
    plt.show()
