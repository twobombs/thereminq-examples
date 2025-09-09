#!/bin/bash

# This script runs the tsp.py script with various parameters in parallel.
# It saves the output of each run into a separate file in the 'results' directory.

# --- Configuration ---
# Get the number of available CPU threads to limit the number of parallel jobs.
MAX_JOBS=$(nproc)
echo "Running with a maximum of $MAX_JOBS parallel jobs."

# Detect the number of available GPUs (for NVIDIA GPUs).
# It redirects errors to /dev/null and defaults to 1 if nvidia-smi is not found.
# Added | head -n 1 to ensure only a single line is read.
N_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1 || echo 1)
echo "Detected $N_GPUS GPUs to cycle through."


# --- Main Execution ---
# Create a directory for the results if it doesn't exist
mkdir -p results

# Set fixed parameters
ITERATIONS=1
# This counter will track the total number of jobs we've launched
total_jobs_launched=0

# Outer loop for nodes (iterating through powers of 2)
for nodes in 32 64 128 256 512
do
  # Loop for quality from 1 to 10
  for quality in $(seq 1 10)
  do
    # Loop for correction_quality from 1 to 10
    for correction_quality in $(seq 1 10)
    do
      # --- Job Management ---
      # If we have already launched the maximum number of jobs,
      # wait for the next one to finish before launching a new one.
      # This keeps the number of running jobs at or below MAX_JOBS.
      if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
      fi

      # --- Job Execution (in background) ---
      # Assign a GPU by cycling through the available ones using the modulo operator.
      current_gpu=$((total_jobs_launched % N_GPUS))

      # Use the 'nodes' variable as the seed
      SEED=$nodes

      # Define the output filename based on the current variables
      OUTPUT_FILE="results/tsp_n${nodes}_q${quality}_cq${correction_quality}_i${ITERATIONS}_s${SEED}.txt"

      # We run the command inside a subshell `()` and in the background `&`
      (
        echo "Starting job (n=$nodes q=$quality cq=$correction_quality) on GPU $current_gpu"
        # Set the GPU context to platform 0 and the specific device, then run the python script.
        export PYOPENCL_CTX="0:${current_gpu}" && python3 tsp.py $nodes $quality $correction_quality $ITERATIONS $SEED > "$OUTPUT_FILE"
      ) &

      # Increment the counter for total jobs launched
      total_jobs_launched=$((total_jobs_launched + 1))
    done
  done
done

# Wait for all remaining background jobs to complete
echo "All jobs have been launched. Waiting for the last running jobs to finish..."
wait

echo "All iterations complete. Results are saved in the 'results' directory."


