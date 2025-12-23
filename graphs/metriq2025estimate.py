import matplotlib.pyplot as plt
import numpy as np

# --- Setup Data ---

# 1. Achieved Systems (Blue - Solid)
# Approx data: (Logical Operations/Depth, Logical Qubits)
achieved_x = [20, 50, 60, 100, 300, 400] 
achieved_y = [20, 53, 72, 127, 433, 1000] # e.g. Sycamore, Zuchongzhi, Eagle, Osprey, Condor
achieved_labels = ["Sycamore", "Zuchongzhi", "Eagle", "Osprey", "Condor", "Atom Computing"]

# 2. Estimated Requirements (Hollow Circles)
# Categories: Supremacy (Low depth), Physics, ML, Finance, Crypto (High depth)
# Note: Crypto is split into "Public Estimate" and "Hidden/Optimized"
est_supremacy_x = [100, 500]
est_supremacy_y = [60, 100]

est_physics_x = [1e3, 1e5]
est_physics_y = [200, 5000]

est_ml_x = [1e4, 1e7]
est_ml_y = [100, 800]

est_finance_x = [1e5, 1e8]
est_finance_y = [1000, 5000]

# The "Public" Crypto Estimate (Shor's classic)
est_crypto_x = [1e11] 
est_crypto_y = [20e6] # 20 Million physical qubits (Old estimate)

# The "Hybrid/Optimized" Crypto Estimate (Newer, lower qubit count, higher depth)
est_crypto_new_x = [1e10]
est_crypto_new_y = [4000] # Logical qubits (Newer estimate)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xscale('log')
ax.set_yscale('log')

# A. The "Old" Simulation Line (State Vector)
ax.axhline(y=53, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Old "Supremacy" Line (53 Qubits)')

# B. The "New" TN/MPS Simulation Frontier (The Rising Floor)
# Logic: TNs can simulate N qubits if depth D is low. 
# As Depth (X) increases, simulatable Qubits (Y) must drop.
# Model: Y = C / X^alpha (roughly)
tn_x = np.logspace(0, 12, 100)
# This curve represents: "Above this line is hard for Tensor Networks"
# For low depth (10-100), we can simulate 1000+ qubits. For high depth, we struggle.
tn_y = 50000 / (tn_x**0.55) 
# Clip the curve so it doesn't go below ~30 qubits (brute force limit)
tn_y = np.maximum(tn_y, 40)

ax.plot(tn_x, tn_y, color='#D32F2F', linestyle='--', linewidth=2, label='Tensor Network Simulation Limit')

# C. The "Bollinger Bandwidth" (The Fog of War)
# Define an "Advantage Horizon" - the likely requirements for useful Quantum Advantage
# This is dropping over time due to hybrid algos, but disappearing into classified zones.
advantage_y = tn_y * 10 # Heuristic: Advantage is 1 order of magnitude above simulation
# Shade the "Squeeze" Zone
ax.fill_between(tn_x, tn_y, advantage_y, color='orange', alpha=0.1, label='The "Squeeze" (Hybrid/Uncertainty Zone)')

# --- Scatter Plots ---

# Achieved
ax.scatter(achieved_x, achieved_y, s=300, c='#2196F3', alpha=0.8, edgecolors='black', label='Achieved (Hardware)')

# Estimated (Standard)
ax.scatter(est_supremacy_x, est_supremacy_y, s=300, facecolors='none', edgecolors='#2196F3', linestyle='--', linewidth=2, label='Est: Supremacy')
ax.scatter(est_physics_x, est_physics_y, s=300, facecolors='none', edgecolors='#00BCD4', linestyle='--', linewidth=2, label='Est: Physics Sim')
ax.scatter(est_finance_x, est_finance_y, s=300, facecolors='none', edgecolors='#F44336', linestyle='--', linewidth=2, label='Est: Finance')
ax.scatter(est_ml_x, est_ml_y, s=300, facecolors='none', edgecolors='#E040FB', linestyle='--', linewidth=2, label='Est: ML')

# Crypto (The "Aaronson Warning")
ax.scatter(est_crypto_x, est_crypto_y, s=400, facecolors='none', edgecolors='#FFC107', linestyle='--', linewidth=2, label='Est: Crypto (Public/Old)')
# The "Hidden" Estimate - ghosted out
ax.scatter(est_crypto_new_x, est_crypto_new_y, s=400, facecolors='none', edgecolors='orange', linestyle=':', linewidth=2, alpha=0.6)
ax.text(1e10, 2500, 'Classified/Hybrid\nCrypto Estimate?', color='orange', ha='center', fontsize=9, alpha=0.8)

# --- Annotations & Styling ---

# Text: The TN Reality
ax.text(1.5, 3000, 'Tensor Networks can simulate\nHigh Qubit counts at Low Depth', color='#D32F2F', fontsize=10, weight='bold')
ax.text(1e8, 20, 'High Depth forces\nSimulation Limit down', color='#D32F2F', fontsize=10, ha='center')

# Text: The Squeeze
ax.text(1e4, 600, 'THE SQUEEZE', color='orange', fontsize=16, weight='bold', alpha=0.5, rotation=-15)

# Axis Labels
ax.set_xlabel('Logical Operations / Circuit Depth (Log Scale)', fontsize=12)
ax.set_ylabel('Logical Qubits (Log Scale)', fontsize=12)
ax.set_title('Quantum Progress: The "Bollinger Squeeze" (TN Limits vs. Hybrid Advantage)', fontsize=16, pad=20)
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend(loc='upper right', frameon=True, fontsize=10)

# Limits
ax.set_xlim(1, 1e12)
ax.set_ylim(10, 1e8)

plt.tight_layout()
plt.show()
