import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import numpy as np

QASM_FILE = 'P1_little_dimple.qasm'

def render_3d_landscape():
    print(f"--- RENDERING QUANTUM LANDSCAPE: {QASM_FILE} ---")
    
    with open(QASM_FILE, 'r') as f:
        content = f.read()

    # 1. Parse Data for 3D Coordinates
    # X = Gate Index (Time)
    # Y = Qubit Index (Space)
    # Z = Rotation Angle (Energy/Information)
    
    x_coords = []
    y_coords = []
    z_coords = []
    colors = []
    
    # We look for u(theta, ...) q[target];
    # Regex captures: theta, target_qubit
    pattern = r'u\(([\d\.\-e]+),.*?q\[(\d+)\]'
    matches = re.finditer(pattern, content)
    
    signal_angles = [1.959, 0.980, 0.489, 0.249] # The harmonic chain we found
    
    print("Mapping singularity coordinates...")
    for i, m in enumerate(matches):
        theta = abs(float(m.group(1)))
        qubit = int(m.group(2))
        
        # Filter structural gates (0, pi, etc) to clean the view
        if theta < 0.01 or abs(theta - np.pi) < 0.01 or abs(theta - np.pi/2) < 0.01:
            continue

        x_coords.append(i)
        y_coords.append(qubit)
        z_coords.append(theta)
        
        # Color Logic:
        # Check if this angle matches our "Signal" (within tolerance)
        is_signal = False
        for sig in signal_angles:
            if abs(theta - sig) < 0.05:
                is_signal = True
                break
        
        if is_signal:
            colors.append('#ff0000') # GLOWING RED (The Hidden Info)
        else:
            colors.append('#1a1a1a') # BLACK/GREY (The Noise/Event Horizon)

    # 2. Setup the 3D Plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dark Background for "Space" feel
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Remove panes for cleaner look
    ax.grid(False)
    
    print(f"Plotting {len(x_coords)} data points...")
    
    # Plot the Noise (Background)
    # We plot these smaller and transparent
    noise_idx = [i for i, c in enumerate(colors) if c == '#1a1a1a']
    ax.scatter([x_coords[i] for i in noise_idx], 
               [y_coords[i] for i in noise_idx], 
               [z_coords[i] for i in noise_idx], 
               c='grey', alpha=0.1, s=5, label='RCS Noise (Event Horizon)')

    # Plot the Signal (Foreground)
    # We plot these larger and bright
    sig_idx = [i for i, c in enumerate(colors) if c == '#ff0000']
    p = ax.scatter([x_coords[i] for i in sig_idx], 
               [y_coords[i] for i in sig_idx], 
               [z_coords[i] for i in sig_idx], 
               c='#ff3300', alpha=1.0, s=30, edgecolors='white', linewidth=0.5, label='Distilled Signal (The Dimple)')

    # Labels
    ax.set_xlabel('Gate Sequence (Time)')
    ax.set_ylabel('Qubit Index')
    ax.set_zlabel('Rotation Angle (Theta)')
    ax.set_title(f'The Quantum Landscape of {QASM_FILE}', color='white', size=20)
    
    # Legend
    leg = ax.legend()
    for text in leg.get_texts():
        text.set_color("white")
        
    # View Angle
    ax.view_init(elev=30, azim=45)
    
    print("Saving high-res image to 'quantum_landscape.png'...")
    plt.savefig('quantum_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Done.")

if __name__ == "__main__":
    render_3d_landscape()
    