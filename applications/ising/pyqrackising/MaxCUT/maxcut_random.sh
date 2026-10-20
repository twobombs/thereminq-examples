#!/bin/bash

# This script runs the tsp.py script with various parameters in parallel.
# It has been updated to be more robust, efficient, and flexible.

# --- Configuration ---
# Usage: ./maxcut_random.sh <nodes> "[quality_sequence]" [optional_seed]
# Example: ./maxcut_random.sh 1024 "$(seq 1 5)" 12345
if [[ -z "$1" ]]; then
    echo "Error: You must provide a single number for nodes."
    echo "Usage: $0 <nodes> \"[quality_range]\" [optional_seed]"
    exit 1
fi

NODE_INPUT=${1}
QUALITY_RANGE=${2:-"1 2 3"}
SEED_OVERRIDE=${3:-""}

if [[ $# -eq 1 ]]; then
    FIXED_SEED_DEFAULT=$RANDOM$RANDOM
    echo "Only nodes argument provided. Using fixed random seed $FIXED_SEED_DEFAULT for all iterations."
    NODE_SEQUENCE=$(seq 16 $NODE_INPUT)
    echo "Will iterate through node counts from 32 to $NODE_INPUT."
else
    NODE_SEQUENCE=$NODE_INPUT
fi


# --- Resource Management ---
MAX_JOBS=$(( $(nproc) / 2 ))
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

for nodes_val in $NODE_SEQUENCE
do
  ITERATIONS=1 
  echo "--- Starting runs for node count $nodes_val ($ITERATIONS iterations) ---"

  for quality in $QUALITY_RANGE
  do
    for i in $(seq 1 $ITERATIONS)
    do
      if [[ -n "$FIXED_SEED_DEFAULT" ]]; then
        SEED="$FIXED_SEED_DEFAULT"
      elif [[ -n "$SEED_OVERRIDE" ]]; then
        SEED="$SEED_OVERRIDE"
      else
        SEED=$((nodes_val * 100000 + quality * 1000 + i))
      fi

      formatted_nodes=$(printf "%04d" $nodes_val)
      
      # --- MODIFICATION START ---
      
      # --- Job 1: Default Execution ---
      if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then wait -n; fi
      
      current_gpu=$((total_jobs_launched % N_GPUS))
      OUTPUT_FILE="results/macxut_n${formatted_nodes}_q${quality}_i${i}_s${SEED}.txt"

      (
        echo "Starting job (n=$nodes_val q=$quality i=$i s=$SEED) [DEFAULT] on GPU $current_gpu"
        export PYOPENCL_CTX="0:${current_gpu}" && python3 maxcut_random.py --n-nodes $nodes_val --quality $quality --seed $SEED > "$OUTPUT_FILE"
      ) &
      total_jobs_launched=$((total_jobs_launched + 1))
      
      # --- Job 2: --is-alt-gpu Execution ---
      if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then wait -n; fi

      current_gpu_alt=$((total_jobs_launched % N_GPUS))
      OUTPUT_FILE_GPU_ON="results/macxut_n${formatted_nodes}_q${quality}_i${i}_s${SEED}_gpu-on.txt"

      (
        echo "Starting job (n=$nodes_val q=$quality i=$i s=$SEED) [--is-alt-gpu] on GPU $current_gpu_alt"
        export PYOPENCL_CTX="0:${current_gpu_alt}" && python3 maxcut_random.py --n-nodes $nodes_val --quality $quality --seed $SEED --is-alt-gpu > "$OUTPUT_FILE_GPU_ON"
      ) &
      total_jobs_launched=$((total_jobs_launched + 1))

      # --- MODIFICATION END ---
    done
  done
done

echo "All jobs have been launched. Waiting for the last running jobs to finish..."
wait

echo "All iterations complete. Results are saved in the 'results' directory."
