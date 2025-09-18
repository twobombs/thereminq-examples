#!/bin/bash

# This script runs the tsp.py script with various parameters in parallel.
# It has been updated to be more robust, efficient, and flexible.

# --- Configuration ---
# MODIFIED: Script now requires a single value for nodes.
# Usage: ./maxcut_random.sh <nodes> "[quality_sequence]" "[correction_quality_sequence]" [optional_seed]
# Example: ./maxcut_random.sh 1024 "$(seq 1 5)" "$(seq 1 5)" 12345
if [[ -z "$1" ]]; then # MODIFIED: Add check to ensure nodes value is provided.
    echo "Error: You must provide a single number for nodes."
    echo "Usage: $0 <nodes> \"[quality_range]\" \"[correction_quality_range]\" [optional_seed]"
    exit 1
fi

NODES=${1}
QUALITY_RANGE=${2:-"$(seq 1 2)"}
CORRECTION_QUALITY_RANGE=${3:-"$(seq 1 2)"}
SEED_OVERRIDE=${4:-""}

# MODIFIED: Use a fixed seed if only the nodes argument is provided.
FIXED_SEED_DEFAULT=""
if [[ $# -eq 1 ]]; then
    FIXED_SEED_DEFAULT=$RANDOM # MODIFIED: Use a random seed for the default
    echo "Only nodes argument provided. Using fixed random seed $FIXED_SEED_DEFAULT for all iterations."
fi

# MODIFIED: Iterations are now always set to the number of nodes.
ITERATIONS=$NODES
echo "Node count is $NODES. Running $ITERATIONS iterations for each parameter combination."

# --- Resource Management ---
MAX_JOBS=$(( $(nproc) / 16 ))
if [[ $MAX_JOBS -lt 1 ]]; then
    MAX_JOBS=1
fi
echo "Running with a maximum of $MAX_JOBS parallel jobs."

if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    echo "Detected $N_GPUS NVIDIA GPUs."
else
    N_GPUS=1
    echo "nvidia-smi not found. Assuming 1 available OpenCL device (e.g., CPU)."
fi

if [[ $N_GPUS -eq 0 && $(command -v nvidia-smi &> /dev/null) ]]; then
    echo "Error: No GPUs detected by nvidia-smi. Exiting."
    exit 1
fi


# --- Main Execution ---
mkdir -p results
total_jobs_launched=0

# The 'nodes' variable is now a single value, so this outer loop only runs once.
for nodes_val in $NODES
do
  for quality in $QUALITY_RANGE
  do
    for correction_quality in $CORRECTION_QUALITY_RANGE
    do
      for i in $(seq 1 $ITERATIONS)
      do
        if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
          wait -n
        fi

        current_gpu=$((total_jobs_launched % N_GPUS))
        
        # MODIFIED: Updated seed logic to prioritize the new fixed seed default.
        if [[ -n "$FIXED_SEED_DEFAULT" ]]; then
          SEED="$FIXED_SEED_DEFAULT"
        elif [[ -n "$SEED_OVERRIDE" ]]; then
          # When a seed is passed, it is now kept exactly the same for all iterations.
          SEED="$SEED_OVERRIDE"
        else
          # Generate a unique seed for each iteration if no override is given
          SEED=$((nodes_val * 100000 + quality * 1000 + correction_quality * 10 + i))
        fi

        formatted_nodes=$(printf "%04d" $nodes_val)
        # Filename still includes iteration 'i' to prevent overwrites
        OUTPUT_FILE="results/macxut_n${formatted_nodes}_q${quality}_cq${correction_quality}_i${i}_s${SEED}.txt"

        (
          echo "Starting job (n=$nodes_val q=$quality cq=$correction_quality i=$i s=$SEED) on GPU $current_gpu"
          export PYOPENCL_CTX="0:${current_gpu}" && python3 maxcut_random.py $nodes_val $quality $correction_quality $SEED > "$OUTPUT_FILE"
        ) &

        total_jobs_launched=$((total_jobs_launched + 1))
      done
    done
  done
done

echo "All jobs have been launched. Waiting for the last running jobs to finish..."
wait

echo "All iterations complete. Results are saved in the 'results' directory."
