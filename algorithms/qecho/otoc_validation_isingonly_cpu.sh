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

# Outer loop: Iterate over n_qubits from 4 to 1024
for (( n_qubits=4; n_qubits<=512; n_qubits++ )); do

  # --- MODIFIED LINE ---
  # Inner loop: Iterate over depth from 4 up to the current n_qubits
  for (( depth=4; depth<=512; depth++ )); do
    
    # Check if our job pool is full.
    if [[ $ACTIVE_JOBS -ge $NUM_CORES ]]; then
      wait -n  # Wait for the next (any) job to finish
      ((ACTIVE_JOBS--))
    fi

    # --- MODIFIED: CPU Load Check ---
    # We have a free job slot, but now we also check the 1-minute
    # system load average. We wait until it's <= (2 * NUM_CORES).
    
    # Get 1-min load avg (e.g., "2.34") from /proc/loadavg
    CURRENT_LOAD=$(awk '{print $1}' /proc/loadavg)
    # Calculate the new maximum allowed load
    MAX_LOAD=$((NUM_CORES * 2))

    # Use 'bc' for floating point comparison.
    # Loop WHILE load > (2 * cores)
    while (( $(echo "$CURRENT_LOAD > $MAX_LOAD" | bc -l) )); do
        echo "Job slot free, but load ($CURRENT_LOAD) > 2x cores ($MAX_LOAD). Waiting 1s..."
        sleep 1 # Wait 1 second before re-checking
        
        # Get fresh load average
        CURRENT_LOAD=$(awk '{print $1}' /proc/loadavg)
    done
    # --- End of MODIFIED logic ---


    # Use printf to format numbers with a leading zero (e.g., 4 -> 04)
    # Note: This padding might be insufficient for numbers > 99
    printf -v n_qubits_padded "%02d" $n_qubits
    printf -v depth_padded "%02d" $depth

    # Define log file name using padded numbers
    LOG_FILE="${LOG_DIR}/q${n_qubits_padded}_d${depth_padded}.log"

    # Print status (using the original, non-padded numbers for clarity)
    echo "Starting (Job $TOTAL_JOBS_LAUNCHED): q=$n_qubits, d=$depth. Logging to $LOG_FILE"
    
    # Launch the job in the background
    (
      
      # --- NEW: Added time command ---
      # We wrap the command in { ... } and redirect its stderr (where 'time'
      # writes its output) to the log file.
      {
        time python3 otoc_validation_isingonly_cpu.py $n_qubits $depth 1 0.25 > $LOG_FILE 2>&1
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
