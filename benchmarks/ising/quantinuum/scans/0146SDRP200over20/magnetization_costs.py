import pandas as pd
import plotly.graph_objects as go
import ast

def create_magnetization_heatmap(file_path):
    """
    Reads log data from a file, processes it, and creates a heatmap
    of average magnetization vs. width and binned seconds.

    The resulting plot is displayed and saved as 'magnetization_vs_seconds.png'.

    Args:
        file_path (str): The path to the log file.
    """
    # --- 1. Read and Parse the Data ---
    # The log file contains dictionary-like strings on each line.
    # We'll read each line and safely evaluate it as a Python dictionary.
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(ast.literal_eval(line.strip()))
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping malformed line: {line.strip()} - Error: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        # As a fallback for demonstration, create some sample data
        data = [
            {'width': 10, 'depth': 1, 'magnetization': 0.8, 'seconds': 1},
            {'width': 10, 'depth': 2, 'magnetization': 0.7, 'seconds': 200},
            {'width': 20, 'depth': 1, 'magnetization': 0.9, 'seconds': 400},
            {'width': 20, 'depth': 2, 'magnetization': 0.85, 'seconds': 600},
        ]
        print("Using sample data for demonstration.")

    # --- 2. Create a Pandas DataFrame ---
    # A DataFrame provides a powerful and flexible way to work with structured data.
    if not data:
        print("No data was loaded. Cannot create heatmap.")
        return

    df = pd.DataFrame(data)
    
    # --- 3. Bin the 'seconds' data and Pivot ---
    # To create a heatmap with a continuous variable like 'seconds', we first
    # group the 'seconds' values into discrete bins.
    try:
        # Create 20 bins for the 'seconds' column.
        # pd.cut divides the range of data into intervals.
        df['seconds_bin'] = pd.cut(df['seconds'], bins=20)

        # Create a pivot table.
        # - The y-axis ('index') will be 'width'.
        # - The x-axis ('columns') will be the binned 'seconds'.
        # - The color of each cell will be the *average* 'magnetization'
        #   for that width and time bin.
        heatmap_data = df.pivot_table(
            index='width',
            columns='seconds_bin',
            values='magnetization',
            aggfunc='mean'
        )

        # Convert column names (which are Interval objects) to strings for display
        heatmap_data.columns = heatmap_data.columns.astype(str)

        # --- 4. Create the Heatmap Figure ---
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='viridis',
            colorbar=dict(title='Avg. Magnetization')
        ))

        # --- 5. Customize the Layout ---
        fig.update_layout(
            title='Magnetization Heatmap vs. Time (Seconds)',
            xaxis_title='Time Bins (Seconds)',
            yaxis_title='Width',
            xaxis=dict(tickangle=-45), # Angle the x-axis labels for readability
            yaxis=dict(type='category'), # Treat width as discrete categories
        )

        # --- 6. Show and Save the Figure ---
        # Display the figure in an interactive window.
        print("Displaying heatmap...")
        fig.show()

        # Save the figure as a PNG file.
        # This requires the 'kaleido' package: pip install kaleido
        try:
            fig.write_image("magnetization_vs_seconds.png")
            print("Successfully saved heatmap as 'magnetization_vs_seconds.png'")
        except Exception as e:
            print(f"\nCould not save the figure. Please make sure you have the 'kaleido' package installed (`pip install kaleido`).")
            print(f"Error: {e}")

    except Exception as e:
        print(f"An error occurred while creating the heatmap: {e}")
        print("Please ensure your data contains 'width', 'seconds', and 'magnetization' columns.")


# --- Execution ---
# To run this, replace 'fullog.txt' with the actual path to your file.
# If the file is in the same directory as the script, the name is sufficient.
create_magnetization_heatmap('fullog.txt')

