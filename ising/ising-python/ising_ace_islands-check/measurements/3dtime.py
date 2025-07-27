import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import ast

# Load and parse the data from fullog.txt
try:
    with open('fullog.txt', 'r') as f:
        # Read all lines from the file
        lines = f.readlines()
    # Use ast.literal_eval to safely parse each line (which is a string representation of a dictionary)
    data = [ast.literal_eval(line.strip()) for line in lines if line.strip()]
    df = pd.DataFrame(data)
except FileNotFoundError:
    print("Error: 'fullog.txt' not found. Please make sure the file is in the correct directory.")
    exit()
except (ValueError, SyntaxError) as e:
    print(f"Error parsing the file 'fullog.txt'. Please ensure it contains valid dictionary-like strings. Error: {e}")
    exit()


# Prepare the data for the 3D plot
# Points are the (width, depth) coordinates
points = df[['width', 'depth']].values
# Values are the computation time in seconds, which we'll put on a log scale for better visualization
values = np.log10(df['seconds'])

# Create a regular grid to interpolate the data onto.
# We'll create a grid of 100x100 points spanning the range of widths and depths.
grid_x, grid_y = np.mgrid[
    df['width'].min():df['width'].max():100j, 
    df['depth'].min():df['depth'].max():100j
]

# Interpolate the log-scaled 'seconds' data onto the grid.
# 'linear' method provides a good balance without creating artifacts.
grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

# Create the 3D plot
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection='3d')

# The core of the plot: plot_surface
# We use a colormap (like 'viridis') to create the heatmap effect.
# The surface is colored based on the Z-values (log of seconds).
surf = ax.plot_surface(
    grid_x, 
    grid_y, 
    grid_z, 
    cmap='viridis',
    edgecolor='none',
    antialiased=True
)

# Set labels for the axes to make the plot understandable
ax.set_xlabel('Width', fontsize=12, labelpad=10)
ax.set_ylabel('Depth', fontsize=12, labelpad=10)
ax.set_zlabel('Computation Time (Log10 of Seconds)', fontsize=12, labelpad=10)

# Set the title for the plot
ax.set_title('3D Log Heatmap of Computation Time vs. Width and Depth', fontsize=16, pad=20)

# Add a color bar to serve as a legend for the heatmap colors
cbar = fig.colorbar(surf, shrink=0.6, aspect=10)
cbar.set_label('Log10(Seconds)', fontsize=12, labelpad=10)

# Improve viewing angle
ax.view_init(elev=20, azim=-120)

# Save the figure to a file
plt.savefig('3d_log_heatmap.png', dpi=300)

# Display the plot in an interactive window on the desktop
print("Displaying interactive 3D plot. Close the plot window to continue.")
plt.show()

print("3D log heatmap has been displayed and saved as '3d_log_heatmap.png'")

