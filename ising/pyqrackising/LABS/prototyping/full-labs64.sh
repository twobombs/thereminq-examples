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
N_START=33
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
MAX_JOBS_PER_DEVICE=2

echo "Successfully detected $NUM_DEVICES devices: ${DEVICES[@]}"
echo "Starting LABS batch run (N=$N_START to $N_END)..."
echo "Enforcing a limit of $MAX_JOBS_PER_DEVICE jobs per device."
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
    # We add `timeout XXXXm` and use `env` to correctly pass the environment variable.
    (
        echo "--- Log Start N=$N (Device $DEVICE_ID) ---"
        # Use `env` to set the variable for the command run by `timeout`
        time timeout 2000m env PYQRACKISING_FPPOW=6 PYOPENCL_CTX=$DEVICE_ID $PYTHON_CMD $N
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
    # The 'timeout' command exits with 124 when it kills a process
    elif [ $exit_code -eq 124 ]; then
        local msg="[Device $DEVICE_ID] KILLED job for N=$N (TIMEOUT > 20m). (Runtime: ${duration}s)"
        echo "$msg" | tee -a $SCRIPT_LOG
        echo "$msg" >> $N_LOG_FILE
    else
        local msg="[Device $DEVICE_ID] ERROR on job for N=$N. (Exit Code: $exit_code) (Runtime: ${duration}s)"
        echo "$msg" | tee -a $SCRIPT_LOG
        echo "$msg" >> $N_LOG_FILE
    fi
}

# --- Main Job Pool Logic (Per-Device Tracking) ---

# We need an associative array to store the PIDs (Process IDs)
# running on each device.
declare -A pids_per_device

# Initialize an empty list for each device
for dev in "${DEVICES[@]}"; do
    pids_per_device[$dev]=""
done

# Loop for each N
for N in $(seq $N_START $N_END); do
    
    # This inner loop will run until it finds a free slot
    slot_found=false
    while [[ "$slot_found" == "false" ]]; do
    
        # Check each device for a free slot
        for DEVICE_ID in "${DEVICES[@]}"; do
            
            # 1. Clean up dead PIDs for this device
            running_pids=""
            for pid in ${pids_per_device[$DEVICE_ID]}; do
                # `kill -0 $pid` checks if the process is still running
                if kill -0 $pid 2>/dev/null; then
                    # PID is still alive, keep it
                    running_pids="$running_pids $pid"
                fi
            done
            pids_per_device[$DEVICE_ID]=$running_pids
            
            # 2. Count active jobs on this device
            # `wc -w` counts the number of "words" (PIDs) in the string
            active_jobs_on_device=$(echo ${pids_per_device[$DEVICE_ID]} | wc -w)
            
            # 3. Check for a free slot
            if [[ $active_jobs_on_device -lt $MAX_JOBS_PER_DEVICE ]]; then
                # Slot found!
                echo "Found free slot on $DEVICE_ID for N=$N ($active_jobs_on_device / $MAX_JOBS_PER_DEVICE busy)"
                
                # Run the job in the background
                run_job $N $DEVICE_ID &
                
                # Get its PID
                new_pid=$!
                
                # Store it in our tracker
                pids_per_device[$DEVICE_ID]="${pids_per_device[$DEVICE_ID]} $new_pid"
                
                # Mark as found and break the inner loops
                slot_found=true
                break
            fi
        done # end of for-each-device
        
        if [[ "$slot_found" == "false" ]]; then
            # No slots free on any device, wait 1 second and check again
            sleep 1
        fi
    done # end of while-slot-not-found
done

# Wait for all remaining background jobs to complete
echo "All jobs (N=$N_START to $N_END) have been launched."
echo "Waiting for final jobs to complete..."
wait

echo "All iterations complete. Check $SCRIPT_LOG and $LOG_DIR/ for results."
