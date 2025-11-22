#!/bin/bash

# ==========================================
# LABS Solver Batch Runner (Optimized)
# ==========================================

# === Configuration ===
SCRIPT_NAME="full-labs.py"
PYTHON_CMD="python3 -u $SCRIPT_NAME" # Added -u for unbuffered output

# Grid Search Settings
LAMBDA_START=0.0
LAMBDA_END=5.0
LAMBDA_STEP=0.1

N_START=4
N_END=31

# Resource Management
MAX_JOBS_PER_DEVICE=2

# Safety & Logging
# Timeout: Send TERM after 2h. If still running 60s later, send KILL.
JOB_TIMEOUT_OPTS="-k 60s 2h" 
SCRIPT_LOG="labs_batch_runner.log"
LOG_DIR="labs_logs"

# === End Configuration ===

# Ensure dot notation for floats
export LC_NUMERIC=C 

# --- Pre-flight Checks ---
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found!"
    exit 1
fi

mkdir -p $LOG_DIR

# --- Dynamic OpenCL Device Detection ---
echo "Detecting available OpenCL GPU devices..."
DETECTED_DEVICES_STR=$(python3 -c "
import sys
try:
    import pyopencl as cl
    devices = []
    for p_idx, platform in enumerate(cl.get_platforms()):
        for d_idx, device in enumerate(platform.get_devices(cl.device_type.GPU)):
            # Returns format PlatformIndex:DeviceIndex
            devices.append(f'{p_idx}:{d_idx}') 
    if not devices:
        print(\"No OpenCL GPU devices found.\", file=sys.stderr)
        exit(1)
    print(' '.join(devices))
except Exception as e:
    print(f\"Error detecting OpenCL devices: {e}\", file=sys.stderr)
    exit(1)
")

if [ $? -ne 0 ]; then
    echo "Device detection failed. Exiting."
    exit 1
fi

DEVICES=($DETECTED_DEVICES_STR)
NUM_DEVICES=${#DEVICES[@]}
echo "Successfully detected $NUM_DEVICES devices: ${DEVICES[@]}"
echo "--- Batch Run Started $(date) ---" > $SCRIPT_LOG 

# --- Safety Trap ---
# Tracks all active PIDs for cleanup
declare -A pids_per_device

cleanup() {
    echo -e "\n!!! Script interrupted. Killing child processes..."
    for dev in "${!pids_per_device[@]}"; do
        for pid in ${pids_per_device[$dev]}; do
            if kill -0 $pid 2>/dev/null; then
                echo "Killing PID $pid on Device $dev..."
                kill -TERM $pid 2>/dev/null
                sleep 0.1
                kill -9 $pid 2>/dev/null
            fi
        done
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- Job Runner Function ---
run_job() {
    local N=$1
    local L_VAL=$2
    local DEVICE_ID=$3
    local JOB_LOG_FILE="$LOG_DIR/labs_run_N_${N}_L_${L_VAL}.log"
    
    echo "[Device $DEVICE_ID] Starting N=$N L=$L_VAL... (Log: $JOB_LOG_FILE)" | tee -a $SCRIPT_LOG
    
    local start_time=$(date +%s)
    
    (
        echo "--- Start: $(date) ---"
        # Pass ENV vars. PYOPENCL_CTX format depends on how python parses it.
        # Assuming your python script handles '0:1' correctly.
        time timeout $JOB_TIMEOUT_OPTS env PYQRACKISING_FPPOW=4 PYOPENCL_CTX=$DEVICE_ID $PYTHON_CMD $N $L_VAL
    ) > "$JOB_LOG_FILE" 2>&1
    
    local exit_code=$?
    local duration=$(( $(date +%s) - start_time ))
    
    if [ $exit_code -eq 0 ]; then
        echo "[Device $DEVICE_ID] Finished N=$N L=$L_VAL. (${duration}s)" | tee -a $SCRIPT_LOG
    elif [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "[Device $DEVICE_ID] TIMEOUT ($JOB_TIMEOUT_OPTS) on N=$N L=$L_VAL." | tee -a $SCRIPT_LOG
    else
        echo "[Device $DEVICE_ID] ERROR (Code $exit_code) on N=$N L=$L_VAL." | tee -a $SCRIPT_LOG
    fi
}

# --- Main Scheduler ---

for dev in "${DEVICES[@]}"; do pids_per_device[$dev]=""; done

for N in $(seq $N_START $N_END); do
    # Use seq to generate, but read into loop to sanitize float format
    for L_RAW in $(seq $LAMBDA_START $LAMBDA_STEP $LAMBDA_END); do
        
        # Sanitize Float: ensures 0.0, 0.1, etc., not 0.30000004
        L_VAL=$(printf "%.1f" "$L_RAW")
        
        slot_found=false
        
        while [[ "$slot_found" == "false" ]]; do
            for DEVICE_ID in "${DEVICES[@]}"; do
                
                # 1. Update PIDs (Filter out dead ones)
                running_pids=""
                current_count=0
                
                # Read current PIDs into array for safer handling
                current_pids_str=${pids_per_device[$DEVICE_ID]}
                
                for pid in $current_pids_str; do
                    if kill -0 $pid 2>/dev/null; then
                        running_pids="$running_pids $pid"
                        ((current_count++))
                    fi
                done
                
                # Trim leading space and update map
                pids_per_device[$DEVICE_ID]=$(echo $running_pids | xargs)
                
                # 2. Check Capacity
                if [[ $current_count -lt $MAX_JOBS_PER_DEVICE ]]; then
                    run_job $N $L_VAL $DEVICE_ID &
                    new_pid=$!
                    
                    # Add new PID to the list
                    pids_per_device[$DEVICE_ID]="${pids_per_device[$DEVICE_ID]} $new_pid"
                    
                    slot_found=true
                    break # Break inner device loop, move to next Lambda
                fi
            done
            
            # If we checked all devices and found no slots, wait before checking again
            if [[ "$slot_found" == "false" ]]; then
                sleep 2
            fi
        done
    done
done

echo "All jobs scheduled. Waiting for remaining jobs to finish..."
wait
echo "Grid search complete."
