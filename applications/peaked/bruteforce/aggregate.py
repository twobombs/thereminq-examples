import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Make sure your CSV has the header: Separability,MaxPagingQB,Duration_Sec,Phase,Measured_Int,Bitstring
FILENAME = 'qrack_results.csv' 

def analyze_results():
    try:
        df = pd.read_csv(FILENAME)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"--- ANALYSIS OF {len(df)} SHOTS ---")
    
    # 1. Top Bitstrings
    print("\n[TOP 5 DETECTED BITSTRINGS]")
    top_strings = df['Bitstring'].value_counts().head(5)
    total_shots = len(df)
    
    for string, count in top_strings.items():
        percent = (count / total_shots) * 100
        print(f"{string} | Count: {count} ({percent:.2f}%)")

    # 2. Phase Clustering
    print("\n[PHASE CLUSTERS]")
    # Round phase to 2 decimal places to group them
    df['Phase_Rounded'] = df['Phase'].round(2)
    phase_counts = df['Phase_Rounded'].value_counts().head(5)
    for phase, count in phase_counts.items():
        print(f"Phase ~ {phase} | Count: {count}")

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bitstring Histogram
    top_10 = df['Bitstring'].value_counts().head(10)
    top_10.plot(kind='bar', ax=ax1, color='cyan')
    ax1.set_title("Top 10 Bitstrings (The Consensus)")
    ax1.set_ylabel("Frequency")
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Phase Histogram
    df['Phase'].plot(kind='hist', bins=50, ax=ax2, color='lime')
    ax2.set_title("Phase Distribution (The Hidden Angle)")
    ax2.set_xlabel("Phase Value")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_results()
