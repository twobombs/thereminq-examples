# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def format_time(seconds):
    """Converts a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365.25)
    
    parts = []
    if years > 0:
        parts.append(f"{years:,.0f} years")
    if days > 0:
        parts.append(f"{days:,.0f} days")
    if hours > 0:
        parts.append(f"{hours:,.0f} hours")
    if minutes > 0:
        parts.append(f"{minutes:,.0f} minutes")
    if seconds > 0:
        parts.append(f"{seconds:.2f} seconds")
        
    if not parts:
        return "0 seconds"
    return ", ".join(parts)

def parse_logs(directory="."):
    """
    Parses all 'qXX_d29.log' files in the given directory.
    """
    file_pattern = re.compile(r'q(\d+)_d29\.log')
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
        print("No valid log files (e.g., 'q04_d29.log') found.")
        return []

    data_points.sort()
    
    # Print all found data
    print("--- All Data Found ---")
    for q, t in data_points:
        print(f"Qubits: {q:<2} -> {t:.3f}s")
    print("----------------------\n")
    
    return data_points

# Define the exponential function
def exp_func(x, a, b):
    return a * np.exp(b * x)

def predict_times_exponential(data, min_qubits_for_fit=17, max_qubits_predict=70):
    """
    Performs an exponential regression on data >= min_qubits_for_fit.
    """
    
    # Filter data to only include the exponential growth part
    exp_data = [(q, t) for q, t in data if q >= min_qubits_for_fit]
    
    if len(exp_data) < 2:
        print(f"Error: Not enough data points (>= {min_qubits_for_fit} qubits) to perform exponential fit.")
        return None

    qubits, times = zip(*exp_data)
    x = np.array(qubits)
    y = np.array(times)

    print(f"--- Exponential Trend Analysis (Using data from {min_qubits_for_fit} qubits) ---")
    
    try:
        # Fit the curve: y = a * e^(b*x)
        # We use p0 as initial guesses. a=1, b=0.5
        params, covariance = curve_fit(exp_func, x, y, p0=[1, 0.5])
    except RuntimeError as e:
        print(f"Error: Could not fit exponential curve. {e}")
        return None
        
    a, b = params
    
    # Calculate R-squared for the exponential fit
    residuals = y - exp_func(x, a, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Model: time = {a:.4e} * e^({b:.4f} * qubits)")
    print(f"R-squared: {r_squared:.6f} (Closer to 1.0 is a better fit)")
    print("------------------------------------------\n")

    # Generate predictions
    target_qubits = np.arange(min(x), max_qubits_predict + 1)
    predicted_times = exp_func(target_qubits, a, b)
    
    return x, y, target_qubits, predicted_times, a, b

def plot_results(all_data, exp_x, exp_y, pred_x, pred_y):
    """Plots the original data and the predicted trend line on a log scale."""
    plt.figure(figsize=(10, 7))
    
    all_q, all_t = zip(*all_data)
    
    # Plot all data points
    plt.scatter(all_q, all_t, color='blue', label='All Actual Data (Qubits 4-23)')
    
    # Plot data used for the fit
    plt.scatter(exp_x, exp_y, color='red', s=100, zorder=5,
                label=f'Data Used for Fit (Qubits {min(exp_x)}-{max(exp_x)})')
    
    # Plot the exponential trend line
    plt.plot(pred_x, pred_y, color='red', linestyle='--', label='Exponential Trend Line')
    
    # Use a logarithmic scale on the Y-axis to see the trend
    plt.yscale('log')
    
    plt.title('Qubit Simulation Time Prediction (Exponential Fit)')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Real Time (seconds) - LOGARITHMIC SCALE')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    plot_filename = 'qubit_time_prediction_exponential.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'")
    plt.show()

def main():
    data = parse_logs(directory="otoc_sweep_log")
    if not data:
        return

    # We start the fit from qubit 17, as that's where the data starts exploding
    fit_start_qubit = 17 
    prediction_result = predict_times_exponential(data, 
                                                    min_qubits_for_fit=fit_start_qubit, 
                                                    max_qubits_predict=70)
    
    if prediction_result is None:
        return
        
    exp_x, exp_y, target_qubits, predicted_times, a, b = prediction_result
    
    print("--- Predicted Times (Based on Exponential Trend) ---")
    
    # Show milestones
    milestones = list(range(fit_start_qubit, 24)) # Show individual qubits near data
    milestones.extend([30, 40, 50, 60, 70])
    
    for q in milestones:
        pred_time = exp_func(q, a, b)
        print(f"  Qubits: {q:<3} -> Predicted Time: {format_time(pred_time)}")
            
    # 4. Plot the results
    plot_results(all_data=data, 
                 exp_x=exp_x, 
                 exp_y=exp_y, 
                 pred_x=target_qubits, 
                 pred_y=predicted_times)

if __name__ == "__main__":
    main()
