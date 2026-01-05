import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import Counter
import os

# --- CONFIGURATION ---
# Replace this with your actual filename
FILENAME = 'P1_little_dimple.qasm' 

# The "Dimple" range you previously identified (for checking density)
DIMPLE_START = 375
DIMPLE_END = 4279

# "Magic" angles to hunt for (The Signal Alphabet)
MAGIC_ANGLES = {
    0: "0",
    np.pi: "pi",
    -np.pi: "-pi",
    np.pi/2: "pi/2",
    -np.pi/2: "-pi/2",
    np.pi/4: "pi/4",
    -np.pi/4: "-pi/4",
    np.pi/8: "pi/8"
}

def parse_qasm_angles(filepath):
    """
    Parses QASM file and extracts all rotation angles (theta).
    Returns a list of tuples: (gate_index, gate_type, angle_value)
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File '{filepath}' not found in the current directory.")
        return []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    angles = []
    # Regex to find gates like u(pi/2, ...) or rz(0.1234)
    pattern = re.compile(r"(\w+)\s*\(([^)]+)\)")
    
    gate_idx = 0
    
    for line in lines:
        if line.strip().startswith("//") or line.strip().startswith("OPENQASM"):
            continue
            
        match = pattern.search(line)
        if match:
            gate_name = match.group(1)
            params_str = match.group(2)
            param_tokens = params_str.split(',')
            
            for token in param_tokens:
                try:
                    # Safe eval for pi handling
                    val_str = token.replace('pi', 'np.pi').strip()
                    val = eval(val_str, {"np": np})
                    # Normalize to -pi to pi range
                    val = (val + np.pi) % (2 * np.pi) - np.pi
                    angles.append((gate_idx, gate_name, val))
                except Exception:
                    pass
            gate_idx += 1
        elif ";" in line: 
            gate_idx += 1
            
    return angles

def analyze_and_plot(data):
    """
    Plots the statistical distribution of angles (Signal vs Noise).
    """
    if not data: return

    indices = [d[0] for d in data]
    thetas = [d[2] for d in data]
    
    print(f"\n--- HAAR DEVIATION ANALYSIS ---")
    print(f"Total Parameterized Gates: {len(thetas)}")
    
    # Filter very small angles (often identity)
    non_zero_thetas = [t for t in thetas if abs(t) > 1e-5]
    print(f"Non-zero rotations: {len(non_zero_thetas)}")

    # Count most common specific angles
    rounded_thetas = [round(t, 4) for t in thetas]
    counts = Counter(rounded_thetas)
    
    print("\n--- Top 5 Repeating Angles (The Signal) ---")
    for angle, count in counts.most_common(5):
        rad_label = f"{angle:.4f}"
        for magic_val, magic_lbl in MAGIC_ANGLES.items():
            if np.isclose(angle, magic_val, atol=1e-3):
                rad_label = f"{magic_lbl} ({angle:.3f})"
        print(f"Angle: {rad_label} | Count: {count}")

    # --- PLOTTING ---
    fig = plt.figure(figsize=(14, 8))
    plt.style.use('dark_background') 

    # PLOT 1: Histogram / KDE
    ax1 = fig.add_subplot(221)
    ax1.hist(thetas, bins=100, density=True, alpha=0.5, color='cyan', label='Angle Counts')
    
    if len(thetas) > 1:
        kde = gaussian_kde(thetas)
        x_grid = np.linspace(min(thetas), max(thetas), 500)
        ax1.plot(x_grid, kde(x_grid), color='yellow', linewidth=2, label='Density (KDE)')

    ax1.set_title("1. Angle Distribution (Signal Detection)")
    ax1.set_xlabel("Rotation Angle (Radians)")
    ax1.legend()
    
    # Highlight magic angles
    for val, lbl in MAGIC_ANGLES.items():
        if min(thetas) <= val <= max(thetas):
            ax1.axvline(x=val, color='white', linestyle=':', alpha=0.3)

    # PLOT 2: Scatter (Depth vs Angle)
    ax2 = fig.add_subplot(222)
    ax2.scatter(indices, thetas, alpha=0.3, s=3, c=thetas, cmap='twilight', marker='.')
    ax2.set_title("2. The Holographic View (Depth vs Angle)")
    ax2.set_xlabel("Gate Index (Time)")
    ax2.set_ylabel("Angle")
    
    return fig # Pass figure to next function if needed

def map_the_skeleton(data):
    """
    Filters for only the 'Crystal' angles and plots their location.
    """
    print("\n--- MAPPING THE SKELETON (Location of Crystal Gates) ---")
    
    # Target signal angles based on your previous run
    targets = {
        "pi/2":  1.5707963267948966,
        "-pi/2": -1.5707963267948966,
        "pi":    3.141592653589793,
        "-pi":   -3.141592653589793
    }
    
    skeleton_indices = []
    
    hits_in_dimple = 0
    
    for idx, gate, angle in data:
        is_hit = False
        for name, val in targets.items():
            if np.isclose(angle, val, atol=1e-4):
                is_hit = True
                break
        
        if is_hit:
            skeleton_indices.append(idx)
            if DIMPLE_START <= idx <= DIMPLE_END:
                hits_in_dimple += 1

    print(f"Total Crystal Gates Found: {len(skeleton_indices)}")
    print(f"Hits inside Dimple Range ({DIMPLE_START}-{DIMPLE_END}): {hits_in_dimple}")
    
    range_len = DIMPLE_END - DIMPLE_START
    if range_len > 0:
        density = hits_in_dimple / range_len * 100
        print(f"Signal Density in Dimple: {density:.4f}%")
        bg_density = (len(skeleton_indices) - hits_in_dimple) / (max(d[0] for d in data) - range_len) * 100
        print(f"Signal Density Elsewhere: {bg_density:.4f}%")
    
    # PLOT 3: The Barcode (Skeleton)
    # We append this to the current figure context if possible, or new one
    plt.subplot(2, 1, 2)
    
    # Create the barcode
    plt.eventplot(skeleton_indices, orientation='horizontal', colors='lime', lineoffsets=1, linelengths=0.8)
    
    # Overlay the Dimple Box
    plt.axvspan(DIMPLE_START, DIMPLE_END, color='red', alpha=0.25, label=f'Dimple Range ({DIMPLE_START}-{DIMPLE_END})')
    
    plt.title("3. The Signal Skeleton: Location of Magic Angles")
    plt.xlabel("Gate Index (Circuit Depth)")
    plt.yticks([])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    extracted_data = parse_qasm_angles(FILENAME)
    if extracted_data:
        analyze_and_plot(extracted_data)
        map_the_skeleton(extracted_data)
