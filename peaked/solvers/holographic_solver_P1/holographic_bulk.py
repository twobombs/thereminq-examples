# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

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

def plot_wormhole_geometry(gates, dimple_indices, title_suffix=""):
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

    # Highlight the Dimple Region if present
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
    ax.set_title(f"Holographic Geometry Analysis {title_suffix}", fontsize=14)
    
    # Invert y axis to match circuit diagrams usually
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- DIMPLE EXTRACTION LOGIC ---

def invert_gate(gate):
    """
    Returns the inverse (dagger) of a single gate dictionary.
    Rule: u(theta, phi, lam)^dagger = u(-theta, -lam, -phi)
    """
    if gate['type'] == 'cz':
        # CZ is its own inverse (Hermitian)
        return f"cz q[{gate['qubits'][0]}],q[{gate['qubits'][1]}];"
    
    elif gate['type'] == 'u':
        # Unpack parameters
        theta, phi, lam = gate['params']
        # Inverse parameters
        # Note: The inverse of U3(θ, �, λ) is U3(-θ, -λ, -�)
        inv_theta = -theta
        inv_phi = -lam  # Swapped and negated
        inv_lam = -phi  # Swapped and negated
        
        return f"u({inv_theta},{inv_phi},{inv_lam}) q[{gate['qubits'][0]}];"
    
    return ""

def generate_extraction_circuits(gates, dimple_indices):
    """
    Generates the U_dagger instructions to strip away the bulk.
    Splits the circuit into: [Pre-Bulk] -> [Dimple] -> [Post-Bulk]
    Returns U_pre_dagger and U_post_dagger.
    """
    if not dimple_indices:
        print("No dimple detected. Cannot generate extraction circuits.")
        return [], []

    start_idx = min(dimple_indices)
    end_idx = max(dimple_indices)
    
    # 1. Isolate the Bulk sections
    pre_bulk_gates = gates[:start_idx]
    post_bulk_gates = gates[end_idx+1:]
    
    print(f"\n--- EXTRACTION PLAN ---")
    print(f"Pre-Dimple Bulk Size:  {len(pre_bulk_gates)} gates")
    print(f"Dimple Size:           {len(dimple_indices)} gates")
    print(f"Post-Dimple Bulk Size: {len(post_bulk_gates)} gates")
    
    # 2. Create U_dagger (Inverse) for Pre-Dimple
    # Reverse order and invert parameters
    u_pre_dagger = [invert_gate(g) for g in reversed(pre_bulk_gates)]
    
    # 3. Create U_dagger (Inverse) for Post-Dimple
    u_post_dagger = [invert_gate(g) for g in reversed(post_bulk_gates)]
    
    # 4. Print Summary
    print("\nTo extract the dimple, wrap the circuit as follows:")
    print("U_dimple = U_post_dagger * U_total * U_pre_dagger")
    
    return u_pre_dagger, u_post_dagger

def save_extracted_layers(gates, dimple_indices, u_pre_dagger, u_post_dagger):
    """
    Saves the 3 circuit layers to QASM files.
    """
    if not dimple_indices: return

    # Define boundaries
    start_idx = min(dimple_indices)
    end_idx = max(dimple_indices)
    
    # 1. Get the Core Dimple (The original gates between the bulk)
    dimple_gates = [g['line'] for g in gates[start_idx : end_idx+1]]
    
    # 2. Write Files
    header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[36];\n'
    
    # Save Pre-Dagger
    with open('layer_1_pre_dagger.qasm', 'w') as f:
        f.write(header)
        f.write('\n'.join(u_pre_dagger))
        print(f"\nSaved 'layer_1_pre_dagger.qasm' ({len(u_pre_dagger)} gates)")

    # Save Dimple Core
    with open('layer_2_dimple_core.qasm', 'w') as f:
        f.write(header)
        f.write('\n'.join(dimple_gates))
        print(f"Saved 'layer_2_dimple_core.qasm' ({len(dimple_gates)} gates)")

    # Save Post-Dagger
    with open('layer_3_post_dagger.qasm', 'w') as f:
        f.write(header)
        f.write('\n'.join(u_post_dagger))
        print(f"Saved 'layer_3_post_dagger.qasm' ({len(u_post_dagger)} gates)")

def verify_extraction():
    """
    Analyzes the extracted 'dimple core' file to visually confirm 
    that the bulk has been removed.
    """
    print("\n--- VERIFYING EXTRACTION ---")
    core_filename = 'layer_2_dimple_core.qasm'
    
    if not os.path.exists(core_filename):
        print(f"Could not find '{core_filename}'. skipping verification.")
        return

    # Analyze the extracted Dimple Core file
    # It should behave like a standalone circuit now
    core_gates, core_dimple_ids = analyze_holographic_geometry(core_filename)
    
    # The core should preserve the Clifford structures
    print(f"Verification: Extracted Core contains {len(core_gates)} gates.")
    
    # Plot the Geometry
    # If successful, the 'Pre' and 'Post' bulk regions (blue blocks at edges) 
    # should be gone. You should see the 'Event Horizon' immediately.
    print("Displaying geometry of the extracted core...")
    plot_wormhole_geometry(core_gates, core_dimple_ids, title_suffix="(Extracted Core)")

# --- EXECUTION ---
if __name__ == "__main__":
    filename = 'P1_little_dimple.qasm'
    
    if not os.path.exists(filename):
        print(f"Error: Please ensure '{filename}' is in the working directory.")
    else:
        # 1. Analyze Original
        gates, dimple_ids = analyze_holographic_geometry(filename)
        
        # 2. Visualize Original
        plot_wormhole_geometry(gates, dimple_ids, title_suffix="(Original)")
        
        # 3. Generate Extraction Circuits
        u_pre, u_post = generate_extraction_circuits(gates, dimple_ids)
        
        # 4. Save Files
        save_extracted_layers(gates, dimple_ids, u_pre, u_post)
        
        # 5. Verify Result
        verify_extraction()
