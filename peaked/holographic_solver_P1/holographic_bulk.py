import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def analyze_holographic_geometry(filename):
    """
    Parses QASM to reveal the 'Hidden Geometry' of the circuit.
    Distinguishes between 'Thermal' (Random) and 'Crystalline' (Clifford/Dimple) gates.
    """
    print(f"Scanning geometry of {filename}...")
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Regex for parsing
    regex_u = re.compile(r"u\(([^,]+),([^,]+),([^)]+)\)\s+q\[(\d+)\];")
    regex_cz = re.compile(r"cz\s+q\[(\d+)\],q\[(\d+)\];")
    
    gates = []
    dimple_indices = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPEN"): continue
        
        # Check for CZ (Entangling / Glue)
        m_cz = regex_cz.match(line)
        if m_cz:
            gates.append({'type': 'cz', 'qubits': [int(m_cz.group(1)), int(m_cz.group(2))], 'line': line})
            continue

        # Check for U (Rotations)
        m_u = regex_u.match(line)
        if m_u:
            # Parse params safely
            raw_params = m_u.groups()[:3]
            params = []
            for p in raw_params:
                try:
                    val = float(p)
                except ValueError:
                    # Handle pi string literals if present
                    val = eval(p.replace('pi', 'np.pi'))
                params.append(val)
            
            # CLASSIFICATION: Is this a "Dimple" (Clifford) or "Bulk" (Random)?
            # Check if all parameters are multiples of pi/2 (Stabilizer operations)
            is_clifford = all(any(np.isclose(abs(val), k * np.pi/2, atol=1e-4) for k in range(5)) for val in params)
            
            q = int(m_u.group(4))
            gates.append({'type': 'u', 'qubits': [q], 'params': params, 'is_dimple': is_clifford, 'line': line})
            
            if is_clifford:
                dimple_indices.append(len(gates)-1)

    print(f"Total Gates: {len(gates)}")
    print(f"Dimple (Clifford) Gates Found: {len(dimple_indices)}")
    
    if len(dimple_indices) > 0:
        print(f"Dimple Location: Gate indices {min(dimple_indices)} to {max(dimple_indices)}")
        print("Dimple Core Sample:")
        for idx in dimple_indices[:5]:
            print(f"  [{idx}] {gates[idx]['line']}")
    
    return gates, dimple_indices

def plot_wormhole_geometry(gates, dimple_indices):
    """
    Visualizes the Circuit as a Spacetime Diagram.
    Red = Dimple (Shockwave), Blue = Bulk (Thermal).
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot background (Bulk)
    for i, g in enumerate(gates):
        color = 'cornflowerblue'
        alpha = 0.1
        if g.get('is_dimple', False):
            color = 'red'
            alpha = 1.0
        elif g['type'] == 'cz':
            color = 'gray' 
            alpha = 0.05
            
        # Draw Gate
        for q in g['qubits']:
            ax.add_patch(
                patches.Rectangle(
                    (i, q - 0.4), # (x, y)
                    1, 0.8,       # width, height
                    color=color,
                    alpha=alpha
                )
            )
            
        # Draw Entanglement Links
        if g['type'] == 'cz':
            q1, q2 = g['qubits']
            ax.plot([i+0.5, i+0.5], [q1, q2], color='black', alpha=0.1, linewidth=0.5)

    # Highlight the Dimple Region
    if dimple_indices:
        start = min(dimple_indices)
        end = max(dimple_indices)
        ax.axvline(x=start, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=end, color='red', linestyle='--', alpha=0.5)
        ax.text(start, 37, "EVENT HORIZON (Entrance)", color='red', fontsize=10, rotation=45)
        ax.text(end, 37, "EVENT HORIZON (Exit)", color='red', fontsize=10, rotation=45)

    ax.set_xlim(0, len(gates))
    ax.set_ylim(-1, 38)
    ax.set_xlabel("Circuit Depth (Time $t$)", fontsize=12)
    ax.set_ylabel("Qubit Index (Space $x$)", fontsize=12)
    ax.set_title("Holographic Disassembly of 'Little Dimple': The Clifford Shockwave", fontsize=14)
    
    # Invert y axis to match circuit diagrams usually
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
filename = 'P1_little_dimple.qasm'
# Note: Ensure the file is in the working directory
try:
    gates, dimple_ids = analyze_holographic_geometry(filename)
    plot_wormhole_geometry(gates, dimple_ids)
except FileNotFoundError:
    print("Please upload the 'P1_little_dimple.qasm' file to run this analysis.")
