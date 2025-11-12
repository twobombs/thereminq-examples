import glob
import re
import math
import vedo
import vedo.pyplot as plt
import numpy as np

# --- Configuration ---
LOG_DIR = "labs_logs"
# --- End Configuration ---


def parse_logs(log_directory):
    """
    Parses all log files in the specified directory.
    
    Returns:
        A list of tuples, sorted by N: [(N, time, cut_value), ...]
    """
    log_files = glob.glob(f"{log_directory}/labs_run_N_*.log")
    
    if not log_files:
        print(f"Error: No log files found in directory '{log_directory}'")
        return []

    print(f"Found {len(log_files)} log files. Parsing...")
    
    data = []
    
    # Regex to find N from the filename
    n_regex = re.compile(r"labs_run_N_(\d+).log")
    # Regex to find the real time
    time_regex = re.compile(r"real\s+(\d+)m([\d.]+)s")
    # Regex to find the cut value
    cut_regex = re.compile(r"Best cut value .*?:\s*([\d.e+-]+)")

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            n_match = n_regex.search(log_file)
            time_match = time_regex.search(content)
            cut_match = cut_regex.search(content)

            if n_match and time_match and cut_match:
                # Extract N
                n = int(n_match.group(1))
                
                # Extract and calculate time in seconds
                minutes = float(time_match.group(1))
                seconds = float(time_match.group(2))
                total_time = (minutes * 60) + seconds
                
                # Extract cut value
                cut_value = float(cut_match.group(1))
                
                data.append((n, total_time, cut_value))
            else:
                print(f"Warning: Could not parse all required data from {log_file}")

        except Exception as e:
            print(f"Warning: Error processing {log_file}: {e}")

    # Sort data by N before returning
    data.sort()
    return data


def plot_3d_scatter_colored(data):
    """
    Visualizes the data in 3D as a filled, colored "ribbon" plot
    in dark mode with an inverted colormap.
    (x=N, y=ln(Time), z=log10(Cut Value))
    """
    if not data:
        print("No data to plot.")
        return

    coords = []
    log_cut_values = []
    for n, time, cut in data:
        # Calculate natural log: ln(Time)
        if time > 0:
            log_time = math.log(time) # Use math.log for natural log
        else:
            log_time = 0 # Placeholder for time=0
        
        # Calculate log10(Cut Value)
        if cut > 0:
            log_cut = math.log10(cut) 
        else:
            log_cut = 0 # Placeholder for cut=0
        
        # Use log_time for the Y coordinate
        coords.append([n, log_time, log_cut])
        log_cut_values.append(log_cut)

    coords_np = np.array(coords)
    log_cut_values_np = np.array(log_cut_values)

    # 1. Create the cloud of points (the tops)
    cloud = vedo.Points(coords_np, r=10).cmap('viridis_r', log_cut_values_np)
    
    # 2. Create the connecting path (the top edge)
    line = vedo.Line(coords_np).lw(4).alpha(1.0).cmap('viridis_r', log_cut_values_np)
    
    # 3. Create the bottom path on the XY plane (z=0)
    start_points = coords_np.copy()
    start_points[:, 2] = 0 # Set all z-values to 0
    
    # 4. Create the two polylines for the ruled surface
    bottom_line = vedo.Line(start_points)
    top_line = vedo.Line(coords_np) 
    
    # 5. Create the "wall" or "ribbon" mesh
    ribbon = vedo.Ribbon(bottom_line, top_line).alpha(0.7)

    # 6. Get the Z-coordinates of all vertices in the ribbon mesh.
    ribbon_z_scalars = ribbon.points[:, 2]
    
    # 7. Apply the scalars and colormap to the ribbon
    ribbon.cmap('viridis_r', ribbon_z_scalars)
    
    # Create the plotter instance
    plotter = vedo.Plotter(
        axes=0, 
        title="", 
        bg='k' 
    )
    
    # Manually add the title
    plotter.add(vedo.Text2D(
        "LABS 3D Run Analysis (Colored by Cut Value)", 
        pos="top-center", 
        c="white", 
        s=1.2
    ))

    # Add custom axes
    axes = vedo.Axes(
        cloud,
        xtitle='N',
        ytitle='ln(Time (s))', # Updated Y-axis title
        ztitle='log10(Cut Value)',
        c='white'
    )

    # Add the color bar legend
    cloud.add_scalarbar(
        title="log10(Cut Value)", 
        c='white', 
        horizontal=False
    )

    plotter.show(cloud, line, ribbon, axes, __doc__)
    plotter.close()


def plot_2d_side_by_side(data):
    """
    Visualizes the data as two 2D scatter plots in dark mode.
    Plot 1: N vs. ln(Time)
    Plot 2: N vs. log10(Cut Value)
    """
    if not data:
        print("No data to plot.")
        return

    # Unpack the data
    n_vals = [d[0] for d in data]
    
    time_vals = []
    for d in data:
        if d[1] > 0: # d[1] is time
            time_vals.append(math.log(d[1])) # Use math.log for natural log
        else:
            time_vals.append(0)
    
    log_cut_vals = []
    for d in data:
        if d[2] > 0:
            log_cut_vals.append(math.log10(d[2])) 
        else:
            log_cut_vals.append(0) 

    # Create a plotter with 2 subplots
    plotter = vedo.Plotter(
        shape=(1, 2), 
        title="", 
        sharecam=False,
        bg='k' 
    )

    # Manually add the title
    plotter.add(vedo.Text2D(
        "LABS 2D Run Analysis", 
        pos="top-center", 
        c="white", 
        s=1.2
    ))

    # --- Plot 1: N vs ln(Time) ---
    # --- MODIFIED (FIX APPLIED) ---
    plot1 = plt.plot(
        n_vals, 
        time_vals, 
        "b-o", 
        xtitle="N", 
        ytitle="ln(Time (s))",
        axes=dict(c='white') # Pass color property here
    )
    # plot1.axes.c('white') # <-- This line was removed (caused error)
    # --- END MODIFIED ---
    plotter.at(0).show(plot1, "N vs.Time")

    # --- Plot 2: N vs log(Cut Value) ---
    # --- MODIFIED (FIX APPLIED) ---
    plot2 = plt.plot(
        n_vals, 
        log_cut_vals, 
        "r-s", 
        xtitle="N", 
        ytitle="log10(Cut Value)",
        axes=dict(c='white') # Pass color property here
    )
    # plot2.axes.c('white') # <-- This line was removed (caused error)
    # --- END MODIFIED ---
    plotter.at(1).show(plot2, "N vs. Cut Value")

    plotter.interactive().close()


def main():
    """
    Main function to parse and plot data.
    """
    all_data = parse_logs(LOG_DIR)
    
    if all_data:
        print(f"Successfully parsed {len(all_data)} log files.")
        
        # --- CHOOSE YOUR VISUALIZATION ---
        
        # Option 1: 3D Scatter Plot with Color Scale
        print("Showing 3D scatter plot with color scale...")
        plot_3d_scatter_colored(all_data)

        # Option 2: 2D Side-by-Side Plots
        print("Showing 2D side-by-side plots...")
        plot_2d_side_by_side(all_data)
        
    else:
        print("No data was parsed. Exiting.")


if __name__ == "__main__":
    main()
