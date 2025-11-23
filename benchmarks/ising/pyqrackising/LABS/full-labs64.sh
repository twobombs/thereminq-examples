#!/bin/bash

# ==========================================
# LABS Solver Batch Runner (v2.0 Optimized)
# ==========================================

# === Configuration ===
SCRIPT_NAME="full-labs.py"
PYTHON_CMD="python3 -u $SCRIPT_NAME" 

# Grid Search Settings
LAMBDA_START=4.0
LAMBDA_END=6.0
LAMBDA_STEP=0.5

N_START=64
N_END=80

# Resource Management
MAX_JOBS_PER_DEVICE=2

# Safety & Logging
TIMEOUT_DURATION="60m"
TIMEOUT_KILL_AFTER="60s"

# Create a unique timestamped directory for this specific batch run
BATCH_ID=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="labs_logs"
LOG_DIR="${LOG_ROOT}/run_${BATCH_ID}"
SCRIPT_LOG="${LOG_DIR}/batch_runner.log"

# === End Configuration ===

# Ensure dot notation
export LC_NUMERIC=C 

# --- Pre-flight Checks ---
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found!"
    exit 1
fi

mkdir -p "$LOG_DIR"

# --- Dynamic OpenCL Device Detection ---
echo "Detecting OpenCL GPU devices..."
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
    print(f\"Error: {e}\", file=sys.stderr)
    exit(1)
")

if [ $? -ne 0 ]; then echo "Device detection failed."; exit 1; fi

DEVICES=($DETECTED_DEVICES_STR)
NUM_DEVICES=${#DEVICES[@]}
echo "Found devices: ${DEVICES[@]}"
echo "Logs will be saved to: $LOG_DIR"

# --- Calculate Total Jobs for Progress Bar ---
# Use bc to calculate floating point range count
L_COUNT=$(echo "($LAMBDA_END - $LAMBDA_START) / $LAMBDA_STEP + 1" | bc)
L_COUNT=${L_COUNT%.*} # Cast to int
N_COUNT=$(( N_END - N_START + 1 ))
TOTAL_JOBS=$(( N_COUNT * L_COUNT ))
CURRENT_JOB=0

# --- Cleanup Trap ---
declare -A pids_per_device

cleanup() {
    echo -e "\n\n[!] Interrupted. Killing active jobs..."
    for dev in "${!pids_per_device[@]}"; do
        pids=$(echo ${pids_per_device[$dev]} | xargs)
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                echo "Killing PID $pid (Dev $dev)"
                kill -TERM $pid 2>/dev/null
                sleep 0.1
                kill -9 $pid 2>/dev/null
            fi
        done
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- Job Runner ---
run_job() {
    local N=$1
    local L_VAL=$2
    local DEVICE_ID=$3
    local LOG_FILE="$LOG_DIR/N${N}_L${L_VAL}.log"
    
    local start_t=$(date +%s)
    
    # Run Command
    (
        time timeout -k $TIMEOUT_KILL_AFTER $TIMEOUT_DURATION \
        env PYQRACKISING_FPPOW=4 PYOPENCL_CTX=$DEVICE_ID \
        $PYTHON_CMD $N $L_VAL
    ) > "$LOG_FILE" 2>&1
    
    local exit_code=$?
    local duration=$(( $(date +%s) - start_t ))
    
    # Log Status
    local status="DONE"
    if [ $exit_code -eq 124 ]; then status="TIMEOUT"; fi
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 124 ]; then status="ERROR"; fi
    
    echo "[$(date +%T)] Dev $DEVICE_ID | $status | N=$N L=$L_VAL (${duration}s)" >> $SCRIPT_LOG
}

# --- Main Loop ---
echo "Starting Batch: $TOTAL_JOBS jobs across $NUM_DEVICES devices." | tee -a $SCRIPT_LOG

for dev in "${DEVICES[@]}"; do pids_per_device[$dev]=""; done

for N in $(seq $N_START $N_END); do
    # Inner loop for floats using python to avoid seq drift or locale issues
    # We generate the list of Lambdas first to iterate cleanly
    LAMBDA_LIST=$(python3 -c "
import numpy as np
for i in np.arange($LAMBDA_START, $LAMBDA_END + $LAMBDA_STEP/2, $LAMBDA_STEP):
    print(f'{i:.1f}')
")
    
    for L_VAL in $LAMBDA_LIST; do
        ((CURRENT_JOB++))
        
        # Visual Progress Bar
        PERCENT=$(( 100 * CURRENT_JOB / TOTAL_JOBS ))
        echo -ne "\rProgress: ${PERCENT}% ($CURRENT_JOB/$TOTAL_JOBS) | Scheduling N=$N L=$L_VAL...   "
        
        slot_found=false
        while [[ "$slot_found" == "false" ]]; do
            for DEVICE_ID in "${DEVICES[@]}"; do
                
                # 1. Refresh PIDs
                running_pids=""
                current_count=0
                for pid in ${pids_per_device[$DEVICE_ID]}; do
                    if kill -0 $pid 2>/dev/null; then
                        running_pids="$running_pids $pid"
                        ((current_count++))
                    fi
                done
                pids_per_device[$DEVICE_ID]=$(echo $running_pids | xargs)
                
                # 2. Launch if slot available
                if [[ $current_count -lt $MAX_JOBS_PER_DEVICE ]]; then
                    run_job $N $L_VAL $DEVICE_ID &
                    pids_per_device[$DEVICE_ID]="${pids_per_device[$DEVICE_ID]} $!"
                    slot_found=true
                    break 
                fi
            done
            
            if [[ "$slot_found" == "false" ]]; then sleep 1; fi
        done
    done
done

echo -e "\nAll jobs scheduled. Waiting for completion..."
wait
echo "Grid search complete. Logs in $LOG_DIR"
