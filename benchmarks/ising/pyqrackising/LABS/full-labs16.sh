#!/bin/bash

# ==========================================
# LABS Solver Batch Runner (Refined)
# ==========================================

# === Configuration ===
PYTHON_CMD="python3 full-labs.py"

# Grid Search Settings
LAMBDA_START=0.0
LAMBDA_END=5.0
LAMBDA_STEP=0.1

N_START=4
N_END=31

# Resource Management
MAX_JOBS_PER_DEVICE=2

# Safety & Logging
# Increased timeout to 2 hours per job (adjust based on N=31 expectations)
JOB_TIMEOUT="2h" 
SCRIPT_LOG="labs_batch_runner.log"
LOG_DIR="labs_logs"

# === End Configuration ===

# Ensure dot notation for floats (0.25 vs 0,25)
export LC_NUMERIC=C 

# --- Dynamic OpenCL Device Detection ---
echo "Detecting available OpenCL GPU devices..."
DETECTED_DEVICES_STR=$(python3 -c "
import sys
try:
    import pyopencl as cl
    devices = []
    for p_idx, platform in enumerate(cl.get_platforms()):
        for d_idx, device in enumerate(platform.get_devices(cl.device_type.GPU)):
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

# --- Safety Trap ---
cleanup() {
    echo -e "\n!!! Script interrupted. Killing child processes..."
    for dev in "${!pids_per_device[@]}"; do
        for pid in ${pids_per_device[$dev]}; do
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null
            fi
        done
    done
    exit 1
}
trap cleanup SIGINT

mkdir -p $LOG_DIR
echo "--- Batch Run Started $(date) ---" > $SCRIPT_LOG 

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
        # Pass environment variables explicitly if needed by the python script
        time timeout $JOB_TIMEOUT env PYQRACKISING_FPPOW=4 PYOPENCL_CTX=$DEVICE_ID $PYTHON_CMD $N $L_VAL
    ) > "$JOB_LOG_FILE" 2>&1
    
    local exit_code=$?
    local duration=$(( $(date +%s) - start_time ))
    
    if [ $exit_code -eq 0 ]; then
        echo "[Device $DEVICE_ID] Finished N=$N L=$L_VAL. (${duration}s)" | tee -a $SCRIPT_LOG
    elif [ $exit_code -eq 124 ]; then
        echo "[Device $DEVICE_ID] TIMEOUT ($JOB_TIMEOUT) on N=$N L=$L_VAL." | tee -a $SCRIPT_LOG
    else
        echo "[Device $DEVICE_ID] ERROR (Code $exit_code) on N=$N L=$L_VAL." | tee -a $SCRIPT_LOG
    fi
}

# --- Main Scheduler ---

declare -A pids_per_device
for dev in "${DEVICES[@]}"; do pids_per_device[$dev]=""; done

for N in $(seq $N_START $N_END); do
    for L_VAL in $(seq $LAMBDA_START $LAMBDA_STEP $LAMBDA_END); do
        
        slot_found=false
        
        while [[ "$slot_found" == "false" ]]; do
            for DEVICE_ID in "${DEVICES[@]}"; do
                
                # 1. Update PIDs (Filter out dead ones)
                running_pids=""
                current_count=0
                for pid in ${pids_per_device[$DEVICE_ID]}; do
                    if kill -0 $pid 2>/dev/null; then
                        running_pids="$running_pids $pid"
                        ((current_count++))
                    fi
                done
                pids_per_device[$DEVICE_ID]=$running_pids
                
                # 2. Check Capacity
                if [[ $current_count -lt $MAX_JOBS_PER_DEVICE ]]; then
                    run_job $N $L_VAL $DEVICE_ID &
                    new_pid=$!
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
