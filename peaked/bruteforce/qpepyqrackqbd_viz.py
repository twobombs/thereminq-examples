import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_csv('qrack_results.csv')

# 2. Prepare Grid (Sorting ensures alignment with Pivot Table)
x_vals = sorted(df['MaxPagingQB'].unique())
y_vals = sorted(df['Separability'].unique())
X, Y = np.meshgrid(x_vals, y_vals)

# 3. Create Pivot Tables for Z-axes
pivot_duration = df.pivot_table(index='Separability', columns='MaxPagingQB', values='Duration_Sec')
pivot_phase = df.pivot_table(index='Separability', columns='MaxPagingQB', values='Phase')

Z_duration = pivot_duration.values
Z_phase = pivot_phase.values

# 4. Generate 3D Plots
fig = plt.figure(figsize=(16, 8))

# --- Plot 1: Duration ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_duration, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_xlabel('Max Paging Qubits')
ax1.set_ylabel('Separability Threshold')
ax1.set_zlabel('Duration (s)')
ax1.set_title('Simulation Duration (Lower is Better)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
ax1.view_init(elev=30, azim=225) # Standard 3D view angle

# --- Plot 2: Phase ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_phase, cmap=cm.inferno, linewidth=0, antialiased=True)
ax2.set_xlabel('Max Paging Qubits')
ax2.set_ylabel('Separability Threshold')
ax2.set_zlabel('Estimated Phase')
ax2.set_title('Phase Estimation (Stability Check)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
ax2.view_init(elev=30, azim=225)

plt.tight_layout()
plt.savefig('qrack_benchmark_3d.png')
plt.show()
