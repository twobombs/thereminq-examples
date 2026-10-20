# -*- coding: utf-8 -*-
# predicting time required to render at 10k qubits and 512 depth from samples found in otoc logs folder

import os
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def format_time(seconds):
    """Converts a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    parts = []
    if days > 0:
        parts.append(f"{int(days)} days")
    if hours > 0:
        parts.append(f"{int(hours)} hours")
    if minutes > 0:
        parts.append(f"{int(minutes)} minutes")
    if seconds > 0:
        parts.append(f"{seconds:.2f} seconds")
        
    return ", ".join(parts)

def parse_logs(directory="."):
    """
    Parses all 'qXX_d512.log' files in the given directory.
    """
    
    # *** MODIFIED LINE ***
    # Updated to look for d512
    file_pattern = re.compile(r'q(\d+)_d512\.log') 
    
    time_pattern = re.compile(r'real\s+(\d+)m([\d\.]+)s')
    
    data_points = []
    scan_path = os.path.abspath(directory)
    print(f"Scanning directory: {scan_path}\n")
    
    try:
        for filename in os.listdir(directory):
            file_match = file_pattern.match(filename)
            if file_match:
                qubits = int(file_match.group(1))
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        time_match = time_pattern.search(content)
                        if time_match:
                            minutes = int(time_match.group(1))
                            seconds = float(time_match.group(2))
                            total_time = (minutes * 60) + seconds
                            data_points.append((qubits, total_time))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {scan_path}")
        return []
    
    if not data_points:
        # *** MODIFIED LINE ***
        print("No valid log files (e.g., 'q04_d512.log') found.")
        return []

    data_points.sort()
    
    print("--- All Data Found ---")
    if len(data_points) > 20:
        print(f"(Showing first 10 and last 10 of {len(data_points)} data points)")
        for q, t in data_points[:10]:
            print(f"Qubits: {q:<3} -> {t:.3f}s")
        print("...")
        for q, t in data_points[-10:]:
            print(f"Qubits: {q:<3} -> {t:.3f}s")
    else:
        for q, t in data_points:
            print(f"Qubits: {q:<3} -> {t:.3f}s")
    print("----------------------\n")
    
    return data_points

def predict_times_linear(data, max_qubits_predict=10000):
    """
    Performs a linear regression on all data.
    """
    if len(data) < 2:
        print("Error: Not enough data points (< 2) to perform regression.")
        return None

    qubits, times = zip(*data)
    x = np.array(qubits)
    y = np.array(times)
    
    result = stats.linregress(x, y)
    
    print("--- Trend Analysis (Linear Regression) ---")
    print(f"Model: time = ({result.slope:.4f} * qubits) + {result.intercept:.4f}")
    print(f"R-squared: {result.rvalue**2:.6f} (Closer to 1.0 is a better fit)")
    print("------------------------------------------\n")
    
    start_qubits = min(x)
    target_qubits = np.arange(start_qubits, max_qubits_predict + 1)
    predicted_times = (result.slope * target_qubits) + result.intercept
    
    return x, y, target_qubits, predicted_times, result.slope, result.intercept

def plot_results_linear(x_data, y_data, x_pred, y_pred):
    """Plots the original data and the predicted trend line."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x_data, y_data, color='blue', label='Actual Data from Logs', s=10, alpha=0.7)
    plt.plot(x_pred, y_pred, color='red', linestyle='--', label='Linear Trend Line')
    
    # *** MODIFIED LINE ***
    plt.title('Qubit Simulation Time Prediction (Linear Fit) - Depth 512')
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Real Time (seconds)')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    plt.xlim(0, max(x_pred) * 1.05) 
    
    # *** MODIFIED LINE ***
    plot_filename = 'qubit_time_prediction_linear_10k_d512.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'")
    plt.show()

def main():
    data = parse_logs(directory="otoc_sweep_log")
    if not data:
        return

    max_qubits = 10000 
    
    prediction_result = predict_times_linear(data, max_qubits_predict=max_qubits)
    
    if prediction_result is None:
        return
        
    x_data, y_data, target_qubits, predicted_times, slope, intercept = prediction_result
    
    print("--- Predicted Times (Based on Linear Trend) ---")
    
    milestones = [70, 200, 500, 1000, 2500, 5000, 7500, 10000]
    
    for q in milestones:
        pred_time = (slope * q) + intercept
        print(f"  Qubits: {q:<5} -> Predicted Time: {format_time(pred_time)}")
            
    plot_results_linear(x_data, y_data, target_qubits, predicted_times)

if __name__ == "__main__":
    main()
