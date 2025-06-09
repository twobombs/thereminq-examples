import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def create_magnetization_heatmap(log_file_path, output_filename="magnetization_heatmap.png"):
    """
    Reads magnetization data from a log file, processes it, creates a heatmap,
    and saves it to a file.

    The function reads a log file where each line is a dictionary-like string.
    It parses this data, creates a pivot table with 'width' as rows, 'depth'
    as columns, and 'magnetization' as values. Finally, it generates, saves,
    and displays a heatmap of the magnetization.

    Args:
        log_file_path (str): The path to the log file to be processed.
        output_filename (str): The filename for the saved PNG image.
    """
    # --- 1. Read and Parse the Data ---
    # We open the log file and read each line.
    # ast.literal_eval is used to safely evaluate the string as a Python literal (in this case, a dictionary).
    try:
        with open(log_file_path, 'r') as f:
            # Each line is read and evaluated into a dictionary, then collected into a list.
            data = [ast.literal_eval(line) for line in f]
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading or parsing the file: {e}")
        return

    # --- 2. Structure the Data with Pandas ---
    # A pandas DataFrame is created from the list of dictionaries for easy manipulation.
    df = pd.DataFrame(data)

    # We pivot the DataFrame to get a 2D grid suitable for a heatmap.
    # 'width' will be the rows (y-axis), 'depth' will be the columns (x-axis),
    # and 'magnetization' will be the values in the cells.
    heatmap_data = df.pivot_table(index='width', columns='depth', values='magnetization')
    
    # Sorting the index (width) is important for a clean, ordered heatmap.
    heatmap_data.sort_index(inplace=True)

    # --- 3. Generate the Heatmap ---
    # We set the size of the figure to ensure the heatmap is readable.
    plt.figure(figsize=(12, 10))

    # seaborn.heatmap is used to create the heatmap.
    # 'annot=False' means the values won't be written on the cells.
    # 'fmt=".3f"' would format the annotation to 3 decimal places if annot were True.
    # 'cmap="viridis"' sets the color map for the heatmap.
    sns.heatmap(heatmap_data, annot=False, cmap="viridis")

    # --- 4. Add Titles and Labels for Clarity ---
    plt.title('Magnetization Heatmap by Width and Depth', fontsize=16)
    plt.xlabel('Depth', fontsize=12)
    plt.ylabel('Width', fontsize=12)
    
    # Ensures that the plot layout is tight and all elements are visible.
    plt.tight_layout()

    # --- 5. Save and Show the Plot ---
    try:
        # Save the figure to a file before showing it.
        # dpi=300 ensures a high-resolution image.
        # bbox_inches='tight' prevents the saved image from being cropped.
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Heatmap successfully saved as '{output_filename}'")
    except Exception as e:
        print(f"An error occurred while saving the heatmap: {e}")
        
    # This command displays the generated heatmap.
    plt.show()

# --- Script Execution ---
# To use this script, replace 'fullog.txt' with the path to your log file.
# The second argument is the name of the output file.
try:
    create_magnetization_heatmap('fullog.txt', 'magnetization_heatmap.png')
except Exception as e:
    print("Could not run with 'fullog.txt'. Ensure the file is in the same directory.")
    # You can create a dummy file to test the script's functionality:
    # with open("dummy_log.txt", "w") as f:
    #     f.write("{'width': 10, 'depth': 1, 'magnetization': 0.8}\n")
    #     f.write("{'width': 10, 'depth': 2, 'magnetization': 0.7}\n")
    #     f.write("{'width': 20, 'depth': 1, 'magnetization': 0.9}\n")
    #     f.write("{'width': 20, 'depth': 2, 'magnetization': 0.85}\n")
    # create_magnetization_heatmap('dummy_log.txt', 'dummy_heatmap.png')

