import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hologram_vis.csv")
plt.scatter(df['x'], df['y'], c=df['amplitude'], cmap='viridis', s=10)
plt.colorbar(label='Density (Amplitude)')
plt.title("Holographic Boundary State (Projected)")
plt.show()
