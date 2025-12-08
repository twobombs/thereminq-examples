#!/bin/bash
# This script runs the Monte Carlo TSP script with various parameters in parallel.
#
# USAGE: ./tsp_monte_carlo.sh [OPTIONAL_SEED]
# If a seed is provided as the first argument, it will be used for all runs.
# Otherwise, a unique seed will be generated for each combination of parameters.

# --- Configuration ---
CLI_SEED=$1 # MODIFIED: Read the first command-line argument for the seed
MULTI_START_RANGE="1"
# K_NEIGHBORS_RANGE is now calculated dynamically based on the number of nodes.
PYTHON_SCRIPT="tsp_monte_carlo.py" # The name of your python script

# --- Resource Management ---
# The divisor controls CPU utilization. 2 = 50%, 1.8 = ~55%, 1 = 100%.
# Using a smaller number increases the number of parallel jobs.
UTILIZATION_DIVISOR="1.8"
MAX_JOBS=$(printf "%.0f" $(echo "$(nproc) / $UTILIZATION_DIVISOR" | bc -l))

if [[ $MAX_JOBS -lt 1 ]]; then
    MAX_JOBS=1
fi
echo "Running with a maximum of $MAX_JOBS parallel jobs."

# --- Main Execution ---
mkdir -p results
total_jobs_launched=0

# Outer loop for nodes, iterating from 32 to 8192 with a step of 2.
for (( nodes=32; nodes<=8192; nodes+=2 ))
do
  # Loop for multi_start
  for multi_start in $MULTI_START_RANGE
  do
    # --- Dynamically set k_neighbors ---
    # Set k_neighbors to the integer closest to the square root of the number of nodes.
    # We use 'bc -l' for floating point math and printf for rounding.
    k_neighbors=$(printf "%.0f" $(echo "sqrt($nodes)" | bc -l))

    # --- Job Management ---
    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
      wait -n
    fi

    # --- MODIFIED: Seed Generation Logic ---
    if [[ -n "$CLI_SEED" ]]; then
      # If a seed was provided via the command line, use it
      SEED="$CLI_SEED"
    else
      # Otherwise, generate a deterministic seed based on parameters
      SEED=$((nodes * 10000 + multi_start * 100 + k_neighbors))
    fi
    # --- End of Modification ---

    # --- Job Execution (in background) ---
    formatted_nodes=$(printf "%04d" $nodes)
    OUTPUT_FILE="results/tspmontecarlo_n${formatted_nodes}_ms${multi_start}_kn${k_neighbors}_s${SEED}.txt"

    (
      echo "Starting job (nodes=$nodes multi_start=$multi_start k_neighbors=$k_neighbors seed=$SEED)"
      python3 "$PYTHON_SCRIPT" "$nodes" "$multi_start" "$k_neighbors" "$SEED" > "$OUTPUT_FILE"
    ) &

    total_jobs_launched=$((total_jobs_launched + 1))
  done
done

echo "All jobs have been launched. Waiting for the last running jobs to finish..."
wait

echo "All iterations complete. Results are saved in the 'results' directory."
