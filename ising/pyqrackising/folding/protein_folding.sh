#!/bin/bash

# Ensure the protein data file exists before starting
CSV_FILE="protein_folding.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: The file $CSV_FILE was not found."
    exit 1
fi

echo "Starting protein folding benchmark..."
echo "======================================="

# Read the CSV file, skipping the header line with tail -n +2
tail -n +2 "$CSV_FILE" | while IFS=',' read -r protein length sequence; do
    # Format the protein data for the command line
    protein_input="$protein,$length,$sequence"
    
    echo ""
    echo "--- Processing Protein: $protein ---"

    # Loop through the specified quality range
    for quality in {2..3}; do
        echo ""
        echo "--> Testing with Quality Level: $quality <--"

        # Run the GPU version
        # echo "Starting GPU run for $protein at quality $quality..."
        # python3 protein_folding.py "$protein_input" "$quality"
        # echo "GPU run for $protein at quality $quality finished."
        
        echo "" # Add a space for readability

        # Run the CPU version
        echo "Starting CPU run for $protein at quality $quality..."
        python3 protein_folding.py "$protein_input" "$quality" cpu
        echo "CPU run for $protein at quality $quality finished."
    done
    echo "--- Finished Processing Protein: $protein ---"
    echo "======================================="
done

echo ""
echo "Benchmark script completed."
