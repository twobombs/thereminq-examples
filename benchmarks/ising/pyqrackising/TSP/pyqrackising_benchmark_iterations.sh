#!/bin/bash

# This script runs the tsp_benchmark_pyqrackising.py script for a range of node sizes.
# It iterates from 2^5 (32) up to 2^16 (65536), calling the benchmark for each size.

echo "Starting TSP Benchmark Suite"
echo "=========================================="

# Ensure the Python script is executable
PYTHON_SCRIPT="pyqrackising_benchmark_iterations.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Benchmark script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# Loop from 2^5 (32) to 2^16 (65536)
for i in $(seq 5 16)
do
  # Calculate the number of nodes for the current iteration (2^i)
  nodes=$((2**i))
  
  echo ""
  echo ">>> Executing benchmark for $nodes nodes..."
  
  # Run the python script, passing the current node size as a command-line argument
  python3 "$PYTHON_SCRIPT" "$nodes"
  
  echo ">>> Finished benchmark for $nodes nodes."
  echo "------------------------------------------"
done

echo ""
echo " Benchmark Suite Complete."
echo "=========================================="
