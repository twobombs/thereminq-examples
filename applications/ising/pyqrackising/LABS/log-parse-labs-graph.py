import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Configuration ===
INPUT_FILE = "labs_results.csv"
OUTPUT_IMG_ENERGY = "plot_energy_scaling.png"
OUTPUT_IMG_PHASE  = "plot_phase_diagram.png"
OUTPUT_IMG_TIME   = "plot_time_scaling.png"

def plot_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run the parser script first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Separate data types
    inf_df = df[df["Is_Inf"] == True]
    valid_df = df[df["Is_Inf"] == False]
    
    # Separate Valid into Trivial (Stuck) and Good
    trivial_df = valid_df[valid_df["Is_Trivial"] == True]
    good_df = valid_df[valid_df["Is_Trivial"] == False]

    # Set graphical style
    sns.set_style("whitegrid")
    print(f"Loaded {len(df)} records. Generating 3 plots...")

    # ==========================================
    # Figure 1: Energy Scaling (Cut vs N)
    # ==========================================
    plt.figure(1, figsize=(10, 6))
    
    # Valid
    plt.scatter(good_df["N"], good_df["Best_Cut"], 
                color="green", alpha=0.7, label="Valid Solution", s=50, edgecolors='k', linewidth=0.5)
    # Trivial
    if not trivial_df.empty:
        plt.scatter(trivial_df["N"], trivial_df["Best_Cut"], 
                    color="orange", marker="s", label="Trivial (Stuck)", s=50, edgecolors='k')

    plt.title("Energy Scaling: Best Cut vs System Size (N)", fontsize=14)
    plt.xlabel("System Size (N)", fontsize=12)
    plt.ylabel("Best Cut Energy", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_ENERGY)

    # ==========================================
    # Figure 2: Phase Diagram (N vs Lambda)
    # ==========================================
    plt.figure(2, figsize=(10, 8))
    
    # Valid (Color mapped)
    if not good_df.empty:
        sc = plt.scatter(good_df["Lambda"], good_df["N"], 
                    c=good_df["Best_Cut"], cmap="viridis", alpha=0.8, s=100, 
                    label="Valid", edgecolors='k', linewidth=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label("Best Cut Energy")
    
    # Trivial (Hollow Square)
    if not trivial_df.empty:
        plt.scatter(trivial_df["Lambda"], trivial_df["N"], 
                    color="none", edgecolor="orange", marker="s", s=150, linewidth=2, 
                    label="Trivial Trap")

    # Infinity (Red X)
    if not inf_df.empty:
        plt.scatter(inf_df["Lambda"], inf_df["N"], 
                    color="red", marker="x", s=100, linewidth=2, 
                    label="Boundary (Inf)")

    plt.title("Phase Diagram: Stability Map (N vs Lambda)", fontsize=14)
    plt.xlabel("Lambda", fontsize=12)
    plt.ylabel("System Size (N)", fontsize=12)
    
    # Add vertical line at 0 for visual clarity
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PHASE)

    # ==========================================
    # Figure 3: Time Scaling (Time vs N)
    # ==========================================
    plt.figure(3, figsize=(10, 6))
    
    # Valid
    plt.scatter(good_df["N"], good_df["Time_Seconds"], 
                color="green", alpha=0.7, label="Valid Solution", s=60, edgecolors='k', linewidth=0.5)

    # Trivial (Often faster or slower depending on trap nature)
    if not trivial_df.empty:
        plt.scatter(trivial_df["N"], trivial_df["Time_Seconds"], 
                    color="orange", marker="s", label="Trivial (Stuck)", s=60, edgecolors='k')

    # Infinity (Did hitting the wall take time?)
    if not inf_df.empty:
        plt.scatter(inf_df["N"], inf_df["Time_Seconds"], 
                    color="red", marker="x", label="Boundary (Inf)", s=60, linewidth=1.5)

    plt.title("Time Complexity: Computation Time vs System Size (N)", fontsize=14)
    plt.xlabel("System Size (N)", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)
    
    # Optional: Log Scale if time varies wildly (uncomment next line if needed)
    # plt.yscale('log') 
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_TIME)
    print(f"Saved Time Plot to {OUTPUT_IMG_TIME}")

    # ==========================================
    # Show UI
    # ==========================================
    print("Opening Graph UI... (Close all windows to exit)")
    plt.show()

if __name__ == "__main__":
    plot_data()
