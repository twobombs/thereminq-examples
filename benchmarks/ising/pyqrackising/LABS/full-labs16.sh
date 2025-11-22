#!/bin/bash

# ==========================================
# LABS Solver Batch Runner (Optimized)
# ==========================================

# === Configuration ===
SCRIPT_NAME="full-labs.py"
# -u forces unbuffered binary stdout/stderr so logs update immediately
PYTHON_CMD="python3 -u $SCRIPT_NAME" 

# Grid Search Settings
LAMBDA_START=0.0
LAMBDA_END=5.0
LAMBDA_STEP=0.25

N_START=4
N_END=31

# Resource Management
MAX_JOBS_PER_DEVICE=2

# Safety & Logging
# syntax: -k <kill_after> <duration>
# Stops job after 2h. If it ignores stop signal, force kills 60s later.
TIMEOUT_DURATION="2h"
TIMEOUT_KILL_AFTER="60s"

SCRIPT_LOG="labs_batch_runner.log"
LOG_DIR="labs_logs"

# === End Configuration ===

# Ensure dot notation for floats (prevents locale issues like 0,25)
export LC_NUMERIC=C 

# --- Pre-flight Checks ---
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found in current directory!"
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
            # Returns format PlatformIndex:DeviceIndex (e.g., 0:0, 0:1)
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

# --- Safety Trap (Cleanup) ---
# Tracks all active PIDs for cleanup to prevent zombies on Ctrl+C
declare -A pids_per_device

cleanup() {
    echo -e "\n\n!!! Script interrupted (SIGINT/SIGTERM). Killing child processes..."
    for dev in "${!pids_per_device[@]}"; do
        # Remove whitespace
        pids=$(echo ${pids_per_device[$dev]} | xargs)
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                echo "Killing PID $pid on Device $dev..."
                # Send terminate then force kill
                kill -TERM $pid 2>/dev/null
                sleep 0.2
                kill -9 $pid 2>/dev/null
            fi
        done
    done
    echo "Cleanup complete. Exiting."
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
        # Pass ENV vars. 
        # PYOPENCL_CTX is set to '0:1' (Platform:Device)
        time timeout -k $TIMEOUT_KILL_AFTER $TIMEOUT_DURATION env PYQRACKISING_FPPOW=4 PYOPENCL_CTX=$DEVICE_ID $PYTHON_CMD $N $L_VAL
    ) > "$JOB_LOG_FILE" 2>&1
    
    local exit_code=$?
    local duration=$(( $(date +%s) - start_time ))
    
    if [ $exit_code -eq 0 ]; then
        echo "[Device $DEVICE_ID] Finished N=$N L=$L_VAL. (${duration}s)" | tee -a $SCRIPT_LOG
    elif [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "[Device $DEVICE_ID] TIMEOUT ($TIMEOUT_DURATION) on N=$N L=$L_VAL." | tee -a $SCRIPT_LOG
    else
        echo "[Device $DEVICE_ID] ERROR (Code $exit_code) on N=$N L=$L_VAL." | tee -a $SCRIPT_LOG
    fi
}

# --- Main Scheduler ---

# Initialize PID tracking
for dev in "${DEVICES[@]}"; do pids_per_device[$dev]=""; done

for N in $(seq $N_START $N_END); do
    # Loop through Lambda values
    for L_RAW in $(seq $LAMBDA_START $LAMBDA_STEP $LAMBDA_END); do
        
        # Float Sanitization: Ensure we get '0.3' instead of '0.3000000004'
        L_VAL=$(printf "%.1f" "$L_RAW")
        
        slot_found=false
        
        # Keep trying until a slot opens on ANY device
        while [[ "$slot_found" == "false" ]]; do
            for DEVICE_ID in "${DEVICES[@]}"; do
                
                # 1. Update PIDs (Filter out dead/finished jobs)
                running_pids=""
                current_count=0
                
                current_pids_str=${pids_per_device[$DEVICE_ID]}
                
                for pid in $current_pids_str; do
                    if kill -0 $pid 2>/dev/null; then
                        running_pids="$running_pids $pid"
                        ((current_count++))
                    fi
                done
                
                # Update the map with only living PIDs
                pids_per_device[$DEVICE_ID]=$(echo $running_pids | xargs)
                
                # 2. Check Capacity
                if [[ $current_count -lt $MAX_JOBS_PER_DEVICE ]]; then
                    # Launch Job in Background
                    run_job $N $L_VAL $DEVICE_ID &
                    new_pid=$!
                    
                    # Add new PID to tracking
                    pids_per_device[$DEVICE_ID]="${pids_per_device[$DEVICE_ID]} $new_pid"
                    
                    slot_found=true
                    break # Break inner loop (Devices), proceed to next Lambda
                fi
            done
            
            # If no slots found on ANY device, wait 2 seconds before checking again
            if [[ "$slot_found" == "false" ]]; then
                sleep 2
            fi
        done
    done
done

echo "All jobs scheduled. Waiting for remaining jobs to finish..."
wait
echo "Grid search complete."
