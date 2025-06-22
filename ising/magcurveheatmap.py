import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# --- Instructions ---
# 1. Create a file named 'fullog.txt' in the same directory as this script.
# 2. Copy and paste your log data into 'fullog.txt'. Each dictionary should be on a new line.
#    Example content for 'fullog.txt':
#    {'width': 96, 'depth': 14, 'trial': 1, 'magnetization': 0.029357910156249997, ...}
#    {'width': 96, 'depth': 15, 'trial': 1, 'magnetization': -0.021504720052083315, ...}
# 3. Run this Python script. It will display the plot and save it as a PNG.

def create_heatmap_from_log(file_path='fullog.txt'):
    """
    Reads quantum simulation log data from a file, processes it,
    and generates a heatmap of magnetization vs. qubit width and circuit depth.
    The heatmap is displayed and saved as a high-resolution PNG.

    Args:
        file_path (str): The path to the log file.
    """
    try:
        # Read the file and parse each line as a dictionary
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                # Use ast.literal_eval to safely parse the string as a Python dictionary
                try:
                    data.append(ast.literal_eval(line.strip()))
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Skipping malformed line: {line.strip()}\nError: {e}")
                    continue

        # Check if any data was successfully parsed
        if not data:
            print("Error: No data was loaded. Please check the content of 'fullog.txt'.")
            return

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns are present
        required_cols = ['width', 'depth', 'magnetization']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: The log file must contain the columns: {', '.join(required_cols)}")
            return

        # Pivot the DataFrame to create a matrix for the heatmap.
        # Index: qubit width, Columns: circuit depth, Values: magnetization.
        # If there are multiple trials for the same width/depth, this will average them.
        heatmap_data = df.pivot_table(index='width', columns='depth', values='magnetization')

        # Create the heatmap using seaborn
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)

        # Add titles and labels for clarity
        plt.title('Magnetization vs. Qubit Width and Circuit Depth', fontsize=16)
        plt.xlabel('Circuit Depth', fontsize=12)
        plt.ylabel('Qubit Width', fontsize=12)

        # Save the figure as a high-resolution PNG file before displaying it
        # dpi=300 ensures high quality for publications or presentations.
        # bbox_inches='tight' removes excess white space around the plot.
        plt.savefig('qubit_magnetization_heatmap.png', dpi=300, bbox_inches='tight')
        print("Heatmap saved as 'qubit_magnetization_heatmap.png'")

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure 'fullog.txt' is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Run the function to generate the heatmap
    create_heatmap_from_log()
