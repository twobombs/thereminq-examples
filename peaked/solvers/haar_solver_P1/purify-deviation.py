import re
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILENAME = 'P1_little_dimple.qasm'
OUTPUT_FILENAME = 'P1_purified.qasm'

# The "Crystal" Alphabet we confirmed
MAGIC_ANGLES = {
    "0": 0.0,
    "pi": np.pi,
    "-pi": -np.pi,
    "pi/2": np.pi/2,
    "-pi/2": -np.pi/2,
    "pi/4": np.pi/4,
    "-pi/4": -np.pi/4,
    "pi/8": np.pi/8  # Keeping pi/8 just in case, common in T-gates
}

def is_magic_angle(val_str):
    """
    Evaluates a string like '0.123' or 'pi/2' and checks if it matches our Crystal set.
    """
    try:
        # Safe eval
        val = eval(val_str.replace('pi', 'np.pi'), {"np": np})
        # Normalize -pi to pi
        val = (val + np.pi) % (2 * np.pi) - np.pi
        
        # Check proximity to any magic angle
        for magic_val in MAGIC_ANGLES.values():
            if np.isclose(val, magic_val, atol=1e-3):
                return True
        return False
    except:
        return False

def purify_circuit(infile, outfile):
    if not os.path.exists(infile):
        print(f"Error: {infile} not found.")
        return

    print(f"Purifying {infile} -> {outfile}...")
    
    with open(infile, 'r') as f:
        lines = f.readlines()
        
    kept_gates = 0
    removed_gates = 0
    
    with open(outfile, 'w') as f:
        for line in lines:
            stripped = line.strip()
            
            # 1. Keep Headers / Comments / Measure / Registers
            if (stripped.startswith("OPENQASM") or 
                stripped.startswith("include") or 
                stripped.startswith("//") or 
                stripped.startswith("qreg") or 
                stripped.startswith("creg") or 
                stripped.startswith("measure") or
                stripped.startswith("barrier")):
                f.write(line)
                continue
            
            # 2. Analyze Gates
            # Regex to separate Gate Name from Params
            # Matches: name(params) targets;  OR  name targets;
            if "(" in stripped:
                # PARAMETERIZED GATE (The suspect)
                # Extract content inside parens
                param_match = re.search(r"\(([^)]+)\)", stripped)
                if param_match:
                    params = param_match.group(1).split(',')
                    
                    # Check if ALL params in the gate are "Magic"
                    # (Usually rotations have 1 param, U3 has 3)
                    all_magic = True
                    for p in params:
                        if not is_magic_angle(p):
                            all_magic = False
                            break
                    
                    if all_magic:
                        f.write(line)
                        kept_gates += 1
                    else:
                        # IT IS NOISE - COMMENT IT OUT
                        # We comment instead of delete to preserve line count structure if needed
                        f.write(f"// NOISE REMOVED: {stripped}\n")
                        removed_gates += 1
                else:
                    # Should not happen if '(' exists, but keep safety
                    f.write(line)
            else:
                # NON-PARAMETERIZED GATE (CX, H, X, Z, etc.)
                # Always keep these to preserve entanglement structure
                f.write(line)
                kept_gates += 1

    print(f"--- PURIFICATION COMPLETE ---")
    print(f"Kept Gates (Skeleton):   {kept_gates}")
    print(f"Removed Gates (Noise):   {removed_gates}")
    print(f"Reduction: {(removed_gates/(kept_gates+removed_gates))*100:.1f}% of computational load removed.")

if __name__ == "__main__":
    purify_circuit(INPUT_FILENAME, OUTPUT_FILENAME)
