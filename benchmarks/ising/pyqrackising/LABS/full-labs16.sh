#!/bin/bash

# ==========================================
# LABS Solver Batch Runner (Grid Search: N & Lambda)
# ==========================================

# === Configuration ===
# 1. PYTHON_CMD: Command to run the script.
PYTHON_CMD="python3 full-labs.py"

# 2. LAMBDA SETTINGS: Iterate Lambda from START to END in STEP increments.
#    Example: 1.0 to 5.0 with step 1.0 = 1.0, 2.0, 3.0, 4.0, 5.0
LAMBDA_START=0.25
LAMBDA_END=5.0
LAMBDA_STEP=0.25

# 3. N SETTINGS: The iteration range for sequence length.
N_START=4
N_END=31

# 4. LOGGING:
SCRIPT_LOG="labs_batch_runner.log"
LOG_DIR="labs_logs"

# 5. MAX_JOBS: How many concurrent Python scripts per GPU?
MAX_JOBS_PER_DEVICE=2
# === End Configuration ===


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
# --- End of Device Detection ---


# --- Safety Trap (Ctrl+C Cleanup) ---
cleanup() {
    echo ""
    echo "!!! Script interrupted (SIGINT). Cleaning up child processes..."
    for dev in "${!pids_per_device[@]}"; do
        for pid in ${pids_per_device[$dev]}; do
            if kill -0 $pid 2>/dev/null; then
                echo " - Killing PID $pid on Device $dev"
                kill -9 $pid 2>/dev/null
            fi
        done
    done
    exit 1
}
trap cleanup SIGINT
# -----------------------------


mkdir -p $LOG_DIR

echo "Successfully detected $NUM_DEVICES devices: ${DEVICES[@]}"
echo "Starting Grid Search:"
echo "  - N: $N_START to $N_END"
echo "  - Lambda: $LAMBDA_START to $LAMBDA_END (Step: $LAMBDA_STEP)"
echo "Logging to $LOG_DIR/"
echo "--- Batch Run Started $(date) ---" > $SCRIPT_LOG 

# Function to run a single job
# Arguments: $1 = N, $2 = LAMBDA, $3 = DEVICE_ID
run_job() {
    local N=$1
    local L_VAL=$2
    local DEVICE_ID=$3
    
    # Log file now includes both N and Lambda to prevent overwriting
    local JOB_LOG_FILE="$LOG_DIR/labs_run_N_${N}_L_${L_VAL}.log"
    
    echo "[Device $DEVICE_ID] Starting N=$N Lambda=$L_VAL... (Log: $JOB_LOG_FILE)" | tee -a $SCRIPT_LOG
    
    local start_time=$(date +%s)
    
    (
        echo "--- Log Start N=$N Lambda=$L_VAL (Device $DEVICE_ID) ---"
        # Pass specific Lambda value to python script
        time timeout 10m env PYQRACKISING_FPPOW=4 PYOPENCL_CTX=$DEVICE_ID $PYTHON_CMD $N $L_VAL
        echo "--- Log End ---"
    ) > $JOB_LOG_FILE 2>&1
    
    local exit_code=$?
    local duration=$(( $(date +%s) - start_time ))
    
    if [ $exit_code -eq 0 ]; then
        local msg="[Device $DEVICE_ID] Finished N=$N L=$L_VAL. (${duration}s)"
        echo "$msg" | tee -a $SCRIPT_LOG
        echo "$msg" >> $JOB_LOG_FILE
    else
        local msg="[Device $DEVICE_ID] ERROR/TIMEOUT on N=$N L=$L_VAL. (Code: $exit_code)"
        echo "$msg" | tee -a $SCRIPT_LOG
        echo "$msg" >> $JOB_LOG_FILE
    fi
}

# --- Main Job Pool Logic ---

declare -A pids_per_device
for dev in "${DEVICES[@]}"; do
    pids_per_device[$dev]=""
done

# OUTER LOOP: N
for N in $(seq $N_START $N_END); do
    
    # INNER LOOP: Lambda
    # We use `seq` to handle floating point steps if needed (e.g. 1.0 0.5 5.0)
    for L_VAL in $(seq $LAMBDA_START $LAMBDA_STEP $LAMBDA_END); do
        
        slot_found=false
        
        # Wait for a free slot
        while [[ "$slot_found" == "false" ]]; do
            
            for DEVICE_ID in "${DEVICES[@]}"; do
                
                # 1. Clean up dead PIDs
                running_pids=""
                for pid in ${pids_per_device[$DEVICE_ID]}; do
                    if kill -0 $pid 2>/dev/null; then
                        running_pids="$running_pids $pid"
                    fi
                done
                pids_per_device[$DEVICE_ID]=$running_pids
                
                # 2. Count active jobs
                active_jobs=$(echo ${pids_per_device[$DEVICE_ID]} | wc -w)
                
                # 3. Check capacity
                if [[ $active_jobs -lt $MAX_JOBS_PER_DEVICE ]]; then
                    
                    # Launch Job
                    run_job $N $L_VAL $DEVICE_ID &
                    new_pid=$!
                    
                    # Track PID
                    pids_per_device[$DEVICE_ID]="${pids_per_device[$DEVICE_ID]} $new_pid"
                    
                    slot_found=true
                    break
                fi
            done 
            
            if [[ "$slot_found" == "false" ]]; then
                sleep 1
            fi
        done
        
    done # End Lambda Loop
done # End N Loop

echo "All jobs launched. Waiting for completion..."
wait
echo "Grid search complete."
