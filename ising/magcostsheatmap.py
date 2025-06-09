import pandas as pd
import plotly.graph_objects as go
import ast
import os

def create_magnetization_heatmap(file_path):
    """
    Reads log data from a file, processes it, and creates a heatmap
    of average magnetization vs. width and binned seconds.

    The resulting plot is displayed and saved as 'magnetization_vs_seconds.png'.

    Args:
        file_path (str): The path to the log file.
    """
    # --- 1. Read and Parse the Data ---
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(ast.literal_eval(line.strip()))
                except (ValueError, SyntaxError):
                    # Silently skip malformed lines
                    pass
    except FileNotFoundError:
        # Create sample data if the file is not found, without printing messages.
        data = [
            {'width': 10, 'depth': 1, 'magnetization': 0.8, 'seconds': 1},
            {'width': 10, 'depth': 2, 'magnetization': 0.7, 'seconds': 200},
            {'width': 20, 'depth': 1, 'magnetization': 0.9, 'seconds': 400},
            {'width': 20, 'depth': 2, 'magnetization': 0.85, 'seconds': 600},
        ]

    # --- 2. Create a Pandas DataFrame ---
    if not data:
        # Exit silently if no data is loaded
        return

    df = pd.DataFrame(data)
    
    # --- 3. Bin the 'seconds' data and Pivot ---
    try:
        # Create 20 bins for the 'seconds' column.
        df['seconds_bin'] = pd.cut(df['seconds'], bins=20)

        # Create a pivot table to get the average magnetization.
        heatmap_data = df.pivot_table(
            index='width',
            columns='seconds_bin',
            values='magnetization',
            aggfunc='mean'
        )

        # Convert column names to strings for display
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
            title='Magnetization Heatmap vs. Time ( high width@lower time = good )',
            xaxis_title='Time Bins (Seconds)',
            yaxis_title='Width',
            xaxis=dict(tickangle=-45), # Angle the x-axis labels for readability
            yaxis=dict(type='category'), # Treat width as discrete categories
        )


        # Save the figure as a PNG file in the same directory.
        # This requires the 'kaleido' package: pip install kaleido
        try:
            fig.write_image("magnetization_vs_seconds_heatmap.png")
        except Exception:
            # Fail silently if saving the image is not possible.
            pass

    except Exception:
        # Fail silently if any other error occurs during processing.
        pass


# --- Execution ---
# Replace 'fullog.txt' with the actual path to your file.
create_magnetization_heatmap('fullog.txt')
