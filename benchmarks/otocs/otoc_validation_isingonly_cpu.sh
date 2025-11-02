#!/bin/bash

# we remove pyopencl to force cpu processing

pip uninstall -y pyopencl

# --- Script Configuration ---
LOG_DIR="otoc_sweep_log"
mkdir -p $LOG_DIR

# --- 1. Auto-Detect CPU Cores ---
echo "Detecting CPU cores..."

# Get the number of CPU cores
NUM_CORES=$(nproc)
if [ -z "$NUM_CORES" ]; then
    echo "Could not detect number of cores, defaulting to 8."
    NUM_CORES=8
fi

echo "Sweep starting with $NUM_CORES parallel CPU jobs."
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

    # Use printf to format numbers with a leading zero (e.g., 4 -> 04)
    printf -v n_qubits_padded "%02d" $n_qubits
    printf -v depth_padded "%02d" $depth

    # Define log file name using padded numbers
    LOG_FILE="${LOG_DIR}/q${n_qubits_padded}_d${depth_padded}.log"

    # Print status (using the original, non-padded numbers for clarity)
    echo "Starting (Job $TOTAL_JOBS_LAUNCHED): q=$n_qubits, d=$depth. Logging to $LOG_FILE"
    
    # Launch the job in the background
    (
      
      # --- â²ï¸ NEW: Added time command ---
      # We wrap the command in { ... } and redirect its stderr (where 'time'
      # writes its output) to the log file.
      {
        time python3 otoc_validation_isingonly_cpu.py $n_qubits $depth 10 0.25 > $LOG_FILE 2>&1
      } 2>> $LOG_FILE
      # --- End of change ---

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
