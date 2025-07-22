# this visualisation takes the cleaned-up output of the ising_ace_depth_series and graphs it
# https://g.co/gemini/share/e13681913255

import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

# --- Data Loading and Parsing ---
# Modify this path if your file is not in the same directory as the script
file_path = "fullog.txt" 

data = []
try:
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            cleaned_line = re.sub(r'\\s*', '', line.strip())
            if cleaned_line:
                try:
                    data_point = ast.literal_eval(cleaned_line)
                    data.append(data_point)
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping line {line_number} due to parsing error: '{line.strip()}' -> {e}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the file is in the same directory as the script, or update the 'file_path' variable.")
    exit()

if not data:
    print("No data was successfully parsed from the file. Please check the file content and format.")
    exit()

df = pd.DataFrame(data)

for col in ['width', 'depth', 'magnetization', 'seconds']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Expected column '{col}' not found in the data.")

df.dropna(subset=['width', 'depth', 'magnetization', 'seconds'], inplace=True)

if df.empty:
    print("DataFrame is empty after cleaning and parsing. No data to plot.")
    exit()

print("--- DataFrame Info ---")
df.info()
print("\n--- DataFrame Head (First 5 rows) ---")
print(df.head())
print("\n--- Unique values for 'width' column ---")
# Ensure widths are integers for proper sorting and display if they are not already
df['width'] = df['width'].astype(int)
unique_widths_sorted = sorted(df['width'].unique())
print(unique_widths_sorted)


# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Magnetization vs. Depth for each Width
plt.figure(figsize=(14, 8))
for width_val in unique_widths_sorted:
    subset = df[df['width'] == width_val]
    plt.plot(subset['depth'], subset['magnetization'], marker='o', linestyle='-', label=f'Width {width_val}')
plt.title('Magnetization vs. Depth for different Widths')
plt.xlabel('Depth')
plt.ylabel('Magnetization')
plt.legend(title='Width', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('magnetization_vs_depth.png')
print("\nGenerated 'magnetization_vs_depth.png'")
# plt.show()

# Plot 2: Computation Time (Seconds) vs. Depth for each Width
plt.figure(figsize=(14, 8))
for width_val in unique_widths_sorted:
    subset = df[df['width'] == width_val]
    plt.plot(subset['depth'], subset['seconds'], marker='o', linestyle='-', label=f'Width {width_val}')
plt.title('Computation Time (Seconds) vs. Depth for different Widths')
plt.xlabel('Depth')
plt.ylabel('Seconds')
plt.legend(title='Width', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('seconds_vs_depth.png')
print("Generated 'seconds_vs_depth.png'")
# plt.show()

# Plot 3: Magnetization vs. Seconds, colored by Width
plt.figure(figsize=(14, 8))
if len(unique_widths_sorted) <= 10:
    colors = plt.cm.get_cmap('tab10', len(unique_widths_sorted))
else:
    colors = plt.cm.get_cmap('viridis', len(unique_widths_sorted)) 

for i, width_val in enumerate(unique_widths_sorted):
    subset = df[df['width'] == width_val]
    plt.scatter(subset['seconds'], subset['magnetization'], label=f'Width {width_val}', color=colors(i), alpha=0.7, s=50)
plt.title('Magnetization vs. Computation Time (Seconds)')
plt.xlabel('Seconds')
plt.ylabel('Magnetization')
plt.legend(title='Width', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('magnetization_vs_seconds.png')
print("Generated 'magnetization_vs_seconds.png'")
# plt.show()

# Plot 4: Average Magnetization vs. Width (with error bars)
plt.figure(figsize=(14, 8))
# Group by width and calculate mean and standard deviation of magnetization
magnetization_by_width = df.groupby('width')['magnetization'].agg(['mean', 'std']).reset_index()
# Ensure widths are sorted for the line plot
magnetization_by_width = magnetization_by_width.sort_values('width')

plt.errorbar(magnetization_by_width['width'], magnetization_by_width['mean'], 
             yerr=magnetization_by_width['std'], fmt='-o', capsize=5,
             label='Mean Magnetization +/- Std Dev')
plt.title('Average Magnetization vs. Width')
plt.xlabel('Width')
plt.ylabel('Average Magnetization')
plt.xticks(unique_widths_sorted) # Ensure all width values are shown as ticks
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('avg_magnetization_vs_width.png')
print("Generated 'avg_magnetization_vs_width.png'")
# plt.show()

# Plot 5: Box Plot of Magnetization by Width
plt.figure(figsize=(14, 8))
# Create a list of magnetization data for each width
boxplot_data = [df[df['width'] == width_val]['magnetization'] for width_val in unique_widths_sorted]
plt.boxplot(boxplot_data, labels=[str(w) for w in unique_widths_sorted], patch_artist=True)
plt.title('Distribution of Magnetization by Width')
plt.xlabel('Width')
plt.ylabel('Magnetization')
plt.grid(True, axis='y') # Grid for y-axis can be helpful for boxplots
plt.tight_layout()
plt.savefig('boxplot_magnetization_vs_width.png')
print("Generated 'boxplot_magnetization_vs_width.png'")
# plt.show()


# Optional: Save the parsed DataFrame to a CSV file
try:
    df.to_csv("parsed_data.csv", index=False)
    print("\nParsed data also saved to 'parsed_data.csv' for your review.")
except Exception as e:
    print(f"\nCould not save CSV: {e}")

print("\n--- Script Finished ---")
print("If plots are not displayed automatically, check for .png files in the script's directory.")
