#!/bin/bash

# === Configuration ===
#
# 1. PYTHON_CMD: The command to run your python script.
PYTHON_CMD="python3 full-labs.py"
#
# 2. SCRIPT_LOG: Log for this batch script's overview.
SCRIPT_LOG="labs_batch_runner.log"
#
# 3. LOG_DIR: Directory to store individual log files for each N.
LOG_DIR="labs_logs"
#
# 4. N_START / N_END: The iteration range.
N_START=4
N_END=80
# === End Configuration ===


# --- Dynamic OpenCL Device Detection ---
echo "Detecting available OpenCL GPU devices..."

# Run a Python one-liner to import pyopencl, list platforms and GPU devices,
# and print them in a space-separated "P:D" format.
DETECTED_DEVICES_STR=$(python3 -c "
import sys
try:
    import pyopencl as cl
    devices = []
    for p_idx, platform in enumerate(cl.get_platforms()):
        # We explicitly ask for GPU devices
        for d_idx, device in enumerate(platform.get_devices(cl.device_type.GPU)):
            devices.append(f'{p_idx}:{d_idx}')
    
    if not devices:
        print(\"No OpenCL GPU devices found.\", file=sys.stderr)
        exit(1)
        
    print(' '.join(devices)) # Output the list to stdout
except ImportError:
    print(\"Error: pyopencl library not found. Please install it (pip install pyopencl)\", file=sys.stderr)
    exit(1)
except Exception as e:
    print(f\"Error detecting OpenCL devices: {e}\", file=sys.stderr)
    exit(1)
")

# Check if the device detection script failed
if [ $? -ne 0 ]; then
    echo "Device detection failed. Exiting."
    echo "Details: $DETECTED_DEVICES_STR"
    exit 1
fi

# Convert the space-separated string of devices into a bash array
DEVICES=($DETECTED_DEVICES_STR)
# --- End of Device Detection ---


# Create the log directory if it doesn't exist
mkdir -p $LOG_DIR

NUM_DEVICES=${#DEVICES[@]}

echo "Successfully detected $NUM_DEVICES devices: ${DEVICES[@]}"
echo "Starting LABS batch run (N=$N_START to $N_END) on $NUM_DEVICES devices..."
echo "Logging batch overview to $SCRIPT_LOG"
echo "Logging individual runs to $LOG_DIR/"
echo "--- Batch Run Started $(date) ---" > $SCRIPT_LOG # Clear/create the script log

# This function runs a single job for a given N on a given Device ID
# It's designed to be run in the background.
run_job() {
    local N=$1
    local DEVICE_ID=$2
    # Define a unique log file for this specific N
    local N_LOG_FILE="$LOG_DIR/labs_run_N_${N}.log"
    
    echo "[Device $DEVICE_ID] Starting job for N=$N... (Log: $N_LOG_FILE)" | tee -a $SCRIPT_LOG
    
    # Get start time
    local start_time=$(date +%s)
    
    # Run the python script, setting the OpenCL context for this specific job.
    # All stdout/stderr is redirected to the unique log file for N.
    (
        echo "--- Log Start N=$N (Device $DEVICE_ID) ---"
        time PYOPENCL_CTX=$DEVICE_ID $PYTHON_CMD $N
        echo "--- Log End N=$N (Device $DEVICE_ID) ---"
    ) > $N_LOG_FILE 2>&1
    
    local exit_code=$?
    
    # Get end time and calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Check if the python command was successful
    if [ $exit_code -eq 0 ]; then
        local msg="[Device $DEVICE_ID] Finished job for N=$N. (Runtime: ${duration}s)"
        echo "$msg" | tee -a $SCRIPT_LOG
        echo "$msg" >> $N_LOG_FILE
    else
        local msg="[Device $DEVICE_ID] ERROR on job for N=$N. (Exit Code: $exit_code) (Runtime: ${duration}s)"
        echo "$msg" | tee -a $SCRIPT_LOG
        echo "$msg" >> $N_LOG_FILE
    fi
}

# --- Main Job Pool Logic ---
#
# We loop from N_START to N_END.
# For each N, we pick a device in round-robin.
# We then launch the job in the background (&).
#
# The 'if' block checks if the number of running jobs (jobs -r -p)
# is greater than or equal to our number of devices.
# If it is, 'wait -n' pauses the script until *any* one of the
# background jobs finishes, freeing up a "slot".
#
# This keeps all devices busy without overloading the system.

for N in $(seq $N_START $N_END); do
    # Pick a device in round-robin fashion
    # (N - N_START) % NUM_DEVICES
    DEVICE_INDEX=$(( (N - N_START) % NUM_DEVICES ))
    DEVICE_ID=${DEVICES[$DEVICE_INDEX]}
    
    # Run the job in the background, passing N and DEVICE_ID
    run_job $N $DEVICE_ID &
    
    # Limit the number of concurrent jobs to the number of devices
    ACTIVE_JOBS=$(jobs -r -p | wc -l)
    if [[ $ACTIVE_JOBS -ge $NUM_DEVICES ]]; then
        # Wait for *any* single job to finish before starting the next
        wait -n
    fi
done

# Wait for all remaining background jobs to complete
echo "All jobs (N=$N_START to $N_END) have been launched."
echo "Waiting for final jobs to complete..."
wait

echo "All iterations complete. Check $SCRIPT_LOG and $LOG_DIR/ for results."
