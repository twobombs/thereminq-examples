#!/bin/bash

# everybody's looking at the ladder 
# https://youtu.be/XzO9jGPtrhc?si=hl7L-93FIRkgsC97

# --- Script Configuration ---
LOG_DIR="otoc_sweep_log"
mkdir -p $LOG_DIR

# --- 1. Auto-Detect OpenCL GPUs ---
echo "Detecting OpenCL devices using 'clinfo'..."

# Check if clinfo is installed
if ! command -v clinfo &> /dev/null; then
    echo "-------------------------------------------------------------"
    echo "Error: 'clinfo' command not found."
    echo "Please install 'clinfo' to auto-detect GPUs."
    echo "On Debian/Ubuntu: sudo apt install clinfo"
    echo "-------------------------------------------------------------"
    exit 1
fi

# --- â¨ MODIFIED LINE: Using gsub for robust parsing ---
# This awk command is safer. It strips all non-digits from the fields.
mapfile -t GPUS < <(clinfo -l | awk '
/Platform #/ { plat_field = $2 } 
/Device #/   { 
  dev_field = $3
  plat_num = plat_field
  dev_num = dev_field
  gsub(/[^0-9]/, "", plat_num)  # Strip non-digits (like #, :) from platform
  gsub(/[^0-9]/, "", dev_num)    # Strip non-digits (like #, :) from device
  if (dev_num != "") {          # Only print if we have a valid device number
    printf("%s:%s\n", plat_num, dev_num)
  }
}
')
# --- End of change ---

# Get the number of GPUs from the array
NUM_GPUS=${#GPUS[@]}
if [ $NUM_GPUS -eq 0 ]; then
  echo "Error: 'clinfo' ran, but no valid OpenCL devices were found."
  exit 1
fi

# Get the number of CPU cores
NUM_CORES=$(nproc)
if [ -z "$NUM_CORES" ]; then
    echo "Could not detect number of cores, defaulting to 8."
    NUM_CORES=8
fi

echo "Sweep starting with $NUM_CORES parallel CPU jobs."
echo "Found $NUM_GPUS OpenCL devices for round-robin dispatch:"
printf "  - %s\n" "${GPUS[@]}"
echo "Logging to: $LOG_DIR"
echo ""

# --- 2. Main Loop & Job Dispatch ---
ACTIVE_JOBS=0
TOTAL_JOBS_LAUNCHED=0

# Outer loop: Iterate over n_qubits from 4 to 69
for (( n_qubits=4; n_qubits<=69; n_qubits++ )); do

  # Inner loop: Iterate over depth from 4 to 29
  for (( depth=4; depth<=29; depth++ )); do
    
    # Check if our job pool is full.
    if [[ $ACTIVE_JOBS -ge $NUM_CORES ]]; then
      wait -n  # Wait for the next (any) job to finish
      ((ACTIVE_JOBS--))
    fi

    # Assign the next GPU in the round-robin sequence
    GPU_INDEX=$(( TOTAL_JOBS_LAUNCHED % NUM_GPUS ))
    CURRENT_GPU=${GPUS[$GPU_INDEX]}

    # Use printf to format numbers with a leading zero (e.g., 4 -> 04)
    printf -v n_qubits_padded "%02d" $n_qubits
    printf -v depth_padded "%02d" $depth

    # Define log file name using padded numbers
    # Example: otoc_sweep_log/q04_d04.log
    LOG_FILE="${LOG_DIR}/q${n_qubits_padded}_d${depth_padded}.log"

    # Print status (using the original, non-padded numbers for clarity)
    echo "Starting (Job $TOTAL_JOBS_LAUNCHED): q=$n_qubits, d=$depth on GPU $CURRENT_GPU. Logging to $LOG_FILE"
    
    # Launch the job in the background
    (
      export PYOPENCL_CTX=$CURRENT_GPU
      python3 otoc_validation_isingonly.py $n_qubits $depth 0.25 0 100 1 > $LOG_FILE 2>&1
    ) &

    # Increment our job counters
    ((ACTIVE_JOBS++))
    ((TOTAL_JOBS_LAUNCHED++))

  done
done

# --- 3. Cleanup ---
echo ""
echo "All $TOTAL_JOBS_LAUNCHED jobs have been launched."
echo "Waiting for the last $ACTIVE_JOBS jobs to finish..."
wait
echo "â Parameter sweep complete. Results are in the '$LOG_DIR' directory."
