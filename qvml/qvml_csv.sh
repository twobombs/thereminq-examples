#!/bin/bash

# This script parses all .txt files in the qvml_results directory,
# extracts key metrics, and prints them in CSV format.

RESULTS_DIR="qvml_results"

# Check if the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory '$RESULTS_DIR' not found."
    exit 1
fi

# Print the CSV header
echo "filename,qubits,depth,cost,tensors,indices"

# Loop through all text files in the directory
for file in "$RESULTS_DIR"/*.txt; do
    # Check if the file is readable
    if [ -r "$file" ]; then
        # --- Extract data from the filename ---
        filename=$(basename "$file")
        qubits=$(echo "$filename" | cut -d'_' -f2)
        # Extract depth and remove the .txt extension
        depth=$(echo "$filename" | cut -d'_' -f4 | cut -d'.' -f1)

        # --- Extract metrics from the file content ---
        # Grep the line with the cost, then use sed to extract only the number
        cost=$(grep 'np.float32' "$file" | sed 's/.*np\.float32(\([0-9.]*\)).*/\1/')

        # Get the last line of the file, which contains tensor and index counts
        last_line=$(tail -n 1 "$file")
        tensors=$(echo "$last_line" | sed 's/.*tensors=\([0-9]*\).*/\1/')
        indices=$(echo "$last_line" | sed 's/.*indices=\([0-9]*\).*/\1/')

        # Print all the extracted data as a single CSV row
        echo "$filename,$qubits,$depth,$cost,$tensors,$indices"
    fi
done
