#!/bin/bash

# This script runs simulations in parallel by automatically detecting
# CPU threads for job slots and GPUs for execution, rotating jobs across the GPUs.

# --- Configuration ---
RESULTS_DIR="qvml_results"

# --- Auto-detect CPU threads ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MAX_JOBS=$(nproc)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MAX_JOBS=$(sysctl -n hw.ncpu)
else
    MAX_JOBS=$(getconf _NPROCESSORS_ONLN)
fi
: "${MAX_JOBS:=4}" # Default to 4 if detection fails

# --- Auto-detect GPUs ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found. This script requires NVIDIA drivers."
    exit 1
fi

# Get the number of GPUs by counting the lines of output from a query.
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | wc -l)

if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "Error: No NVIDIA GPUs were detected by the script."
    exit 1
fi

# --- Script Body ---
mkdir -p "$RESULTS_DIR"

echo "Detected $MAX_JOBS CPU threads to use for parallel jobs."
echo "Detected $NUM_GPUS GPUs. Will rotate jobs across GPUs 0 to $(($NUM_GPUS - 1))."
echo "Starting QVML simulations..."

job_count=0
for qubits in $(seq 4 24); do
    for depth in $(seq 4 $qubits); do
        while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
            sleep 1
        done

        gpu_id=$((job_count % NUM_GPUS))
        FILENAME="${RESULTS_DIR}/qubits_${qubits}_depth_${depth}.txt"
        
        echo "Launching: qubits=$qubits, depth=$depth on GPU $gpu_id"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python3 qvml.py "$qubits" "$depth" > "$FILENAME" &

        ((job_count++))
    done
done

echo "Waiting for all remaining jobs to finish..."
wait

echo "All simulations have finished successfully."
