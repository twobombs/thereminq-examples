#!/bin/bash

# This script runs the tsp.py script with various parameters in parallel.
# It has been updated to be more robust, efficient, and flexible.

# --- Configuration ---
# Use command-line arguments for parameter ranges with sensible defaults.
# Usage: ./tsp.sh "[nodes_list]" "[quality_sequence]" "[correction_quality_sequence]"
# Example: ./tsp.sh "32 64 128" "$(seq 1 5)" "$(seq 1 5)"
NODES_TO_RUN=${1:-"$(seq 32 2 4069)"}
QUALITY_RANGE=${2:-"$(seq 1 2)"}
CORRECTION_QUALITY_RANGE=${3:-"$(seq 1 2)"}
ITERATIONS=1 # Set fixed parameter for iterations

# --- Resource Management ---
# Set MAX_JOBS to use all but one available CPU thread, leaving overhead for the system.
# This is much more efficient than the previous conservative setting.
MAX_JOBS=$(( $(nproc) / 2 ))
# Ensure MAX_JOBS is at least 1, even on a single-core machine.
if [[ $MAX_JOBS -lt 1 ]]; then
    MAX_JOBS=1
fi
echo "Running with a maximum of $MAX_JOBS parallel jobs."

# Detect NVIDIA GPUs more robustly.
if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    echo "Detected $N_GPUS NVIDIA GPUs."
else
    # Default to 1 for a non-GPU or non-NVIDIA setup (e.g., CPU-based OpenCL).
    N_GPUS=1
    echo "nvidia-smi not found. Assuming 1 available OpenCL device (e.g., CPU)."
fi

# Exit if nvidia-smi was found but reported no GPUs.
if [[ $N_GPUS -eq 0 && $(command -v nvidia-smi &> /dev/null) ]]; then
    echo "Error: No GPUs detected by nvidia-smi. Exiting."
    exit 1
fi


# --- Main Execution ---
# Create a directory for the results if it doesn't exist
mkdir -p results

# This counter will track the total number of jobs we've launched
total_jobs_launched=0

# Outer loop for nodes (iterating through the specified list)
for nodes in $NODES_TO_RUN
do
  # Loop for quality
  for quality in $QUALITY_RANGE
  do
    # Loop for correction_quality
    for correction_quality in $CORRECTION_QUALITY_RANGE
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

      # Create a unique, reproducible seed for each parameter combination.
      # This ensures that each experimental run is statistically independent.
      SEED=$((nodes * 10000 + quality * 100 + correction_quality))

      # Define the output filename based on the current variables
      OUTPUT_FILE="results/tsp_n${nodes}_q${quality}_cq${correction_quality}_i${ITERATIONS}_s${SEED}.txt"

      # We run the command inside a subshell `()` and in the background `&`
      (
        echo "Starting job (n=$nodes q=$quality cq=$correction_quality s=$SEED) on GPU $current_gpu"
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

