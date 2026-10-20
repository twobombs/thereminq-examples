import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from pathlib import Path
from collections import defaultdict

# No global style is set here, so the default is white.

def parse_log_file(filepath):
    """Extracts solution time and cut weight from a log file."""
    try:
        content = filepath.read_text()
        time_match = re.search(r"Seconds to solution: ([\d.]+)", content)
        weight_match = re.search(r"Cut weight: ([\d.]+)", content)
        time = float(time_match.group(1)) if time_match else None
        weight = float(weight_match.group(1)) if weight_match else None
        return time, weight
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None

def create_2d_plots(results):
    """Generates and saves 2D bar charts comparing the results."""
    valid_results = [r for r in results if r['default_time'] is not None and r['gpu_on_time'] is not None]
    if not valid_results:
        print("\nCould not generate 2D plots due to missing data.")
        return

    labels = [r['name'] for r in valid_results]
    default_times = [r['default_time'] for r in valid_results]
    gpu_on_times = [r['gpu_on_time'] for r in valid_results]
    default_weights = [r['default_weight'] for r in valid_results]
    gpu_on_weights = [r['gpu_on_weight'] for r in valid_results]

    x = np.arange(len(labels))
    width = 0.35

    # Plot 1: Solution Times
    fig_time, ax_time = plt.subplots(figsize=(15, 8))
    ax_time.bar(x - width/2, default_times, width, label='Default Run')
    ax_time.bar(x + width/2, gpu_on_times, width, label='--is-alt-gpu Run')
    ax_time.set_ylabel('Seconds to Solution (s)')
    ax_time.set_title('Comparison of Solution Times')
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=90)
    ax_time.legend()
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)
    fig_time.tight_layout()
    plt.savefig('comparison_times.png')
    print("\nSaved 2D time comparison plot to 'comparison_times.png'")
    plt.show()

    # Plot 2: Cut Weights
    fig_weight, ax_weight = plt.subplots(figsize=(15, 8))
    ax_weight.bar(x - width/2, default_weights, width, label='Default Run')
    ax_weight.bar(x + width/2, gpu_on_weights, width, label='--is-alt-gpu Run')
    ax_weight.set_ylabel('Cut Weight')
    ax_weight.set_title('Comparison of Solution Quality (Cut Weight)')
    ax_weight.set_xticks(x)
    ax_weight.set_xticklabels(labels, rotation=90)
    ax_weight.legend()
    ax_weight.grid(axis='y', linestyle='--', alpha=0.7)
    fig_weight.tight_layout()
    plt.savefig('comparison_weights.png')
    print("Saved 2D weight comparison plot to 'comparison_weights.png'")
    plt.show()

def create_3d_plots(results):
    """Generates interactive 3D surface plots of the results with legends and colorbars."""
    print("\nGenerating 3D surface plots...")
    
    params = set()
    data = defaultdict(dict)
    for r in results:
        match = re.search(r'n(\d+)_q(\d+)', r['name'])
        if match:
            nodes, quality = int(match.group(1)), int(match.group(2))
            params.add((nodes, quality))
            data[(nodes, quality)] = r

    if len(params) < 3:
        print("Not enough data points to generate meaningful 3D plots. Try a wider range of nodes and quality.")
        return

    nodes_vals = sorted(list(set(p[0] for p in params)))
    quality_vals = sorted(list(set(p[1] for p in params)))
    X, Y = np.meshgrid(nodes_vals, quality_vals)
    
    Z_dt = np.full(X.shape, np.nan)
    Z_gt = np.full(X.shape, np.nan)
    Z_dw = np.full(X.shape, np.nan)
    Z_gw = np.full(X.shape, np.nan)

    for i, q in enumerate(quality_vals):
        for j, n in enumerate(nodes_vals):
            if (n, q) in data:
                res = data[(n, q)]
                Z_dt[i, j] = res['default_time']
                Z_gt[i, j] = res['gpu_on_time']
                Z_dw[i, j] = res['default_weight']
                Z_gw[i, j] = res['gpu_on_weight']

    default_color_proxy = Patch(color=plt.cm.viridis(0.5), alpha=0.7, label='Default Run')
    gpu_on_color_proxy = Patch(color=plt.cm.plasma(0.5), alpha=0.7, label='--is-alt-gpu Run')

    # --- Plot 1: 3D Solution Times ---
    fig_time_3d = plt.figure(figsize=(12, 8))
    ax = fig_time_3d.add_subplot(111, projection='3d')
    
    surf1 = ax.plot_surface(X, Y, Z_dt, cmap='viridis', alpha=0.7, rstride=1, cstride=1)
    ax.plot_surface(X, Y, Z_gt, cmap='plasma', alpha=0.7, rstride=1, cstride=1)

    ax.set_xlabel('Node Count')
    ax.set_ylabel('Quality Parameter')
    ax.set_zlabel('Time (s)')
    ax.set_title('3D Surface Plot of Solution Time')
    ax.legend(handles=[default_color_proxy, gpu_on_color_proxy], loc='upper left')
    fig_time_3d.colorbar(surf1, ax=ax, shrink=0.6, aspect=10, label='Time (s)')
    plt.show()

    # --- Plot 2: 3D Cut Weights ---
    fig_weight_3d = plt.figure(figsize=(12, 8))
    ax2 = fig_weight_3d.add_subplot(111, projection='3d')

    surf2 = ax2.plot_surface(X, Y, Z_dw, cmap='viridis', alpha=0.7, rstride=1, cstride=1)
    ax2.plot_surface(X, Y, Z_gw, cmap='plasma', alpha=0.7, rstride=1, cstride=1)
    
    ax2.set_xlabel('Node Count')
    ax2.set_ylabel('Quality Parameter')
    ax2.set_zlabel('Cut Weight')
    ax2.set_title('3D Surface Plot of Cut Weight')
    ax2.legend(handles=[default_color_proxy, gpu_on_color_proxy], loc='upper left')
    fig_weight_3d.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, label='Cut Weight')
    plt.show()

def main():
    """Main function to find, parse, compare, and plot benchmark results."""
    results_dir = Path("results")
    if not results_dir.is_dir():
        print("Error: 'results' directory not found.")
        return

    default_logs = sorted([f for f in results_dir.glob("*.txt") if not f.name.endswith("_gpu-on.txt")])
    if not default_logs:
        print("No result files found.")
        return

    results = []
    for default_path in default_logs:
        gpu_on_path = default_path.with_name(default_path.stem + "_gpu-on.txt")
        default_time, default_weight = parse_log_file(default_path)
        gpu_on_time, gpu_on_weight = parse_log_file(gpu_on_path)
        results.append({
            "name": default_path.stem.replace('macxut_', ''),
            "default_time": default_time, "default_weight": default_weight,
            "gpu_on_time": gpu_on_time, "gpu_on_weight": gpu_on_weight,
        })

    # --- Print Table ---
    print("-" * 110)
    print(f"{'Benchmark Parameters':<45} | {'Default Run':^28} | {'--is-alt-gpu Run':^28}")
    print(f"{'':<45} | {'Time (s)':>12} | {'Cut Weight':>12} | {'Time (s)':>12} | {'Cut Weight':>12}")
    print("-" * 110)
    for res in results:
        dt = f"{res['default_time']:.4f}" if res['default_time'] is not None else "N/A"
        dw = f"{res['default_weight']:.4f}" if res['default_weight'] is not None else "N/A"
        gt = f"{res['gpu_on_time']:.4f}" if res['gpu_on_time'] is not None else "N/A"
        gw = f"{res['gpu_on_weight']:.4f}" if res['gpu_on_weight'] is not None else "N/A"
        print(f"{res['name']:<45} | {dt:>12} | {dw:>12} | {gt:>12} | {gw:>12}")
    print("-" * 110)

    # --- Create and Show Plots ---
    create_2d_plots(results)
    
    # MODIFICATION: Use a style context to apply dark theme only to 3D plots
    with plt.style.context('dark_background'):
        create_3d_plots(results)

if __name__ == "__main__":
    main()
