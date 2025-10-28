import re
import ast
import pandas as pd
from vedo import Points, Plotter, settings
from pathlib import Path
import math

# --- Configuration ---
LOG_DIR = "otoc_sweep_log"
OUTPUT_FILE = "otoc_sweep_3d_plot.png"
# --- End Configuration ---

def parse_log_file(file_path):
    """
    Parses a single log file to find qubits, depth, and log(real time).
    """
    # 1. Parse filename for qubits and depth
    match = re.search(r"q(\d+)_d(\d+)\.log", file_path.name)
    if not match:
        return None
    
    qubits = int(match.group(1))
    depth = int(match.group(2))
    
    # 2. Parse file content for 'real' time
    try:
        content = file_path.read_text()
        
        # --- NEW TIME PARSING LOGIC ---
        # Search for the 'real' time string, e.g., "real	4m19.984s"
        time_match = re.search(r"real\s+(\d+)m([\d\.]+)s", content)
        
        if not time_match:
            print(f"Warning: Could not find 'real' time string in {file_path.name}")
            return None
            
        # Convert minutes and seconds to total seconds
        try:
            minutes = float(time_match.group(1))
            seconds = float(time_match.group(2))
            total_seconds = (minutes * 60) + seconds
        except (ValueError, TypeError) as e:
            print(f"Error converting time string in {file_path.name}: {e}")
            return None

        # Calculate log time
        if total_seconds <= 0:
            print(f"Warning: Non-positive time {total_seconds}s in {file_path.name}")
            return None
            
        log_time = math.log(total_seconds)
        # --- END NEW TIME PARSING LOGIC ---
            
        return {"qubits": qubits, "depth": depth, "log_time": log_time}
        
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None

def main():
    log_path = Path(LOG_DIR)
    if not log_path.is_dir():
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        print("Please run this script in the same folder as your 'otoc_sweep_log' directory.")
        return

    print(f"Scanning {LOG_DIR} for log files...")
    
    data = []
    log_files = list(log_path.glob("q*_d*.log"))
    
    if not log_files:
        print(f"Error: No log files found in '{LOG_DIR}'.")
        return

    for file in log_files:
        parsed_data = parse_log_file(file)
        if parsed_data:
            data.append(parsed_data)
            
    if not data:
        print("Error: No data could be parsed from log files.")
        return
        
    print(f"Successfully parsed {len(data)} log files.")

    # 3. Create Pandas DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by=["qubits", "depth"])

    print("Generating 3D interactive plot...")

    # --- Vedo Plotting Logic (Updated for Log Time) ---

    # 4. Create 3D points
    coords = df[['qubits', 'depth', 'log_time']].values  # Use log_time for Z
    scalars = df['log_time'].values                     # Use log_time for color

    pts = Points(coords, r=8)
    
    pts.pointdata["log_time"] = scalars                 # Use log_time for data
    pts.cmap("viridis", "log_time")                     # Map color to log_time
    
    pts.add_scalarbar(title="Log(Execution Time)", pos=((0.85, 0.1), (0.9, 0.9)))

    # 5. Create a Plotter instance
    plt = Plotter(
        title="OTOC Sweep: Log(Time) vs. Qubits and Depth", # Updated title
        axes={
            'xtitle': 'Number of Qubits',
            'ytitle': 'Depth',
            'ztitle': 'Log(Execution Time)' # Updated Z-axis label
        }
    )

    # 6. Add points to the plotter
    plt.add(pts)

    # 7. Save to PNG
    try:
        print(f"Attempting to save screenshot to: {OUTPUT_FILE}...")
        plt.screenshot(OUTPUT_FILE)
        print(f"Successfully saved screenshot to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Could not save screenshot: {e}")

    # 8. Show the interactive window
    print("Displaying plot in separate window...")
    plt.show() # This opens a native window
    
    print("Done.")

if __name__ == "__main__":
    main()
