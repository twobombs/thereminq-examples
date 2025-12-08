#!/bin/bash

# This script runs the maxcut_benchmarks.py script for a range of sizes,
# scaling up in powers of two from 32 to 65536.

# Define the sequence of graph sizes to test.
# WARNING: Sizes above 4096 may take a very long time to complete.
SIZES=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)

echo "Starting Max-Cut scaling benchmark..."
echo "This will run tests for the following sizes: ${SIZES[*]}"
echo "A separate CSV file will be generated for each run."

# Loop through each size in the SIZES array
for SIZE in "${SIZES[@]}"; do
    echo ""
    echo "========================================"
    echo "Running benchmark for size: $SIZE"
    echo "========================================"
    
    # Execute the python script, passing the current size as a command-line argument.
    # The output of the python script will be displayed here.
    python3 maxcut_benchmarks.py --sizes "$SIZE"
    
    # Check the exit code of the last command.
    # If it's not 0, an error occurred.
    if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Error running benchmark for size $SIZE."
        echo "Aborting the script."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "Scaling benchmark completed successfully."
echo "========================================"
