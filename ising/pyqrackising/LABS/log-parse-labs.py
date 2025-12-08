import os
import re
import pandas as pd
import glob

# === Configuration ===
LOG_DIR = "labs_logs"    
OUTPUT_FILE = "labs_results.csv"

def is_trivial(sequence):
    """
    Detects if a sequence is suspiciously uniform (stuck in local minima).
    Returns True if >90% of bits are identical.
    """
    if not sequence: return False
    
    total = len(sequence)
    if total == 0: return False
    
    zeros = sequence.count('0')
    ones = sequence.count('1')
    
    if (zeros / total > 0.9) or (ones / total > 0.9):
        return True
    return False

def parse_logs():
    data = []
    # Ensure path handling covers different OS separators if needed, though glob usually handles it
    log_files = glob.glob(f"{LOG_DIR}/**/*.log", recursive=True)
    
    print(f"Found {len(log_files)} log files. Parsing...")

    for filepath in log_files:
        filename = os.path.basename(filepath)
        if "batch_runner" in filename: continue

        # Capture N and Lambda (handling negative floats for Lambda)
        match_params = re.search(r"N(\d+)_L(-?\d+(?:\.\d+)?)", filename)
        
        if not match_params:
            continue

        n_val = int(match_params.group(1))
        try:
            l_val = float(match_params.group(2).rstrip('.'))
        except ValueError:
            continue

        best_cut = None
        duration = 0.0
        sequence = ""
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    # === FIX IS HERE ===
                    # Added '+' to the character class [-\d\.eE+] to capture "e+12" style exponents
                    if "Best cut:" in line:
                        cut_match = re.search(r"Best cut:\s+([-\d\.eE+]+|inf|INF)", line, re.IGNORECASE)
                        if cut_match:
                            val_str = cut_match.group(1).lower()
                            best_cut = float('inf') if "inf" in val_str else float(val_str)
                    
                    # Parse duration (handling "XmYs" or just "Xs")
                    if line.strip().startswith("real"):
                        time_match = re.search(r"real\s+(\d+)m([\d\.]+)s", line)
                        if time_match:
                            duration = (float(time_match.group(1)) * 60) + float(time_match.group(2))
                        else:
                            sec_match = re.search(r"real\s+([\d\.]+)s", line)
                            if sec_match:
                                duration = float(sec_match.group(1))
                    
                    if "Original Sequence" in line:
                        seq_match = re.search(r"Original Sequence.*?:\s+([01]+)", line)
                        if seq_match:
                            sequence = seq_match.group(1)

            if best_cut is not None:
                data.append({
                    "N": n_val,
                    "Lambda": l_val,
                    "Best_Cut": best_cut,
                    "Time_Seconds": duration,
                    "Is_Trivial": is_trivial(sequence),
                    "Is_Inf": (best_cut == float('inf')),
                    "Sequence_Preview": sequence[:15] + "..." if len(sequence) > 15 else sequence
                })
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Create and Save
    df = pd.DataFrame(data)
    
    if not df.empty:
        df_csv = df.sort_values(by=["N", "Lambda"])
        df_csv.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nSuccess! Parsed {len(df)} entries.")
        print(f"Results saved to {OUTPUT_FILE}")
        
        # Summary
        valid_df = df[df["Is_Inf"] == False]
        
        print(f"\n=== Scan Summary ===")
        print(f"Total Files: {len(df)}")
        
        if not valid_df.empty:
            # Filter N>=20 for the console print out (optional)
            view_df = valid_df[valid_df["N"] >= 20] if not valid_df[valid_df["N"] >= 20].empty else valid_df
            
            print(f"\n--- Valid Energies (Sample) ---")
            print(view_df.sort_values(by=["N", "Best_Cut"], ascending=[False, False])
                  [["N", "Lambda", "Best_Cut", "Is_Trivial"]].head(10).to_string(index=False))
    else:
        print("No valid data found.")

if __name__ == "__main__":
    parse_logs()
