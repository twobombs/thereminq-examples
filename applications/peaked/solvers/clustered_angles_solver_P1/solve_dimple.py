# solution was deemed incorrect 
#
import re
import numpy as np

QASM_FILE = 'P1_little_dimple.qasm'

def solve():
    print(f"--- SOLVING {QASM_FILE} (CLUSTERING STRATEGY) ---")
    
    with open(QASM_FILE, 'r') as f:
        content = f.read()

    # 1. Extract all rotation angles (Theta)
    # We look for the first parameter of u(...) gates
    matches = re.findall(r'u\(([\d\.\-e]+),', content)
    
    angles = []
    for m in matches:
        val = abs(float(m))
        # Filter out structural gates (0, pi/2, pi, etc.) to isolate the encoded signal
        is_structural = False
        for struct in [0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
            if abs(val - struct) < 0.05: # 0.05 rad tolerance
                is_structural = True
                break
        if not is_structural:
            angles.append(val)

    print(f"analyzed {len(angles)} signal gates.")

    # 2. Perform Clustering (The Fix)
    # Instead of looking for exact matches, we group angles within 2% tolerance.
    # This overcomes the "dimple" noise/obfuscation.
    sorted_angles = sorted(angles)
    clusters = []
    
    if not sorted_angles:
        print("No signal angles found!")
        return

    current_cluster = [sorted_angles[0]]
    TOLERANCE = 0.02 # 2% tolerance allows for the "dimple" variation

    for angle in sorted_angles[1:]:
        mean_val = sum(current_cluster) / len(current_cluster)
        if abs(angle - mean_val) < TOLERANCE:
            current_cluster.append(angle)
        else:
            clusters.append(current_cluster)
            current_cluster = [angle]
    clusters.append(current_cluster)

    # 3. Find the Dominant Signal
    # The largest cluster that isn't structural noise is our answer.
    significant_clusters = [c for c in clusters if len(c) > 10]
    significant_clusters.sort(key=len, reverse=True)

    if not significant_clusters:
        print("No hidden signal found.")
        return

    top_cluster = significant_clusters[0]
    dominant_angle = sum(top_cluster) / len(top_cluster)
    
    print(f"\n--- SIGNAL DETECTED ---")
    print(f"Dominant Angle: {dominant_angle:.6f} rad")
    print(f"Cluster Size:   {len(top_cluster)} repetitions")

    # 4. Decode to Bitstring
    # Phase = Angle / 2pi
    phase_fraction = dominant_angle / (2 * np.pi)
    print(f"Encoded Phase:  {phase_fraction:.8f}")

    print("\n--- DECODED FLAG ---")
    binary_string = ""
    remainder = phase_fraction
    
    # Decode 36 bits
    for _ in range(36):
        remainder *= 2
        if remainder >= 1:
            binary_string += "1"
            remainder -= 1
        else:
            binary_string += "0"
            
    print(binary_string)
    
    # Verify validity
    if len(top_cluster) > 40:
        print("\n[SUCCESS] Signal strength is high. This is the flag.")
    else:
        print("\n[WARNING] Signal is weak. Result might be noise.")

if __name__ == "__main__":
    solve()
    
