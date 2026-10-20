#!/bin/bash
# ---
# Revised script for running Transverse Field Ising Model (TFIM) simulations.
#
# Key Improvements:
# 1. Resumable: Checks for existing, non-empty log files and skips them.
# 2. Max Utilization: Uses ALL available CPU threads for parallelism.
# 3. GPU Load Balancing: Distributes the high volume of threads across available 
#    GPUs using randomized load balancing (oversubscription allowed).
# 4. Enhanced Output: Provides a summary of completed, skipped, and total jobs.
# 5. Granular & Parallel CSVs: Generates a separate summary CSV for each 'T'.
# ---

# --- Configuration ---
PYTHON_SCRIPT="sqr_magnetisation-iterations_cli.py"
OUTPUT_DIR="tfim_results_final"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting TFIM batch simulations..."
echo "Output logs will be saved in: $OUTPUT_DIR"
echo "---"

# --- Detect available CPU threads (Target Parallelism) ---
if command -v nproc &> /dev/null; then
    AVAILABLE_THREADS=$(nproc)
else
    AVAILABLE_THREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)
fi

# Ensure at least 1 job
if [ "$AVAILABLE_THREADS" -lt 1 ]; then AVAILABLE_THREADS=1; fi

# Set parallelism to utilize all threads
PARALLEL_JOBS=$AVAILABLE_THREADS

echo "Detected available CPU threads: $AVAILABLE_THREADS"
echo "Targeting $PARALLEL_JOBS concurrent simulations."
echo "---"

# --- Device Detection Strategy ---
# We use a small python snippet to detect OpenCL GPUs
DETECT_SCRIPT="
import pyopencl as cl
try:
    platforms = cl.get_platforms()
    gpus = []
    for p_idx, platform in enumerate(platforms):
        # Only look for GPUs
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        for d_idx, device in enumerate(devices):
            gpus.append(f'{p_idx}:{d_idx}')
    print(' '.join(gpus))
except:
    print('')
"

# Check if pyopencl is installed and detect devices
if python3 -c "import pyopencl" 2>/dev/null; then
    DETECTED_GPUS=$(python3 -c "$DETECT_SCRIPT")
else
    DETECTED_GPUS=""
fi

if [ -n "$DETECTED_GPUS" ]; then
    # -- GPU MODE --
    export GPU_LIST="$DETECTED_GPUS"
    export MODE="GPU"
    
    GPU_COUNT=$(echo "$DETECTED_GPUS" | wc -w)
    echo "Detected $GPU_COUNT OpenCL GPU(s): $DETECTED_GPUS"
    echo "Load Balancing Strategy: Distributing $PARALLEL_JOBS threads across $GPU_COUNT GPUs."
else
    # -- CPU MODE --
    export MODE="CPU"
    echo "No OpenCL GPUs detected (or PyOpenCL missing). Running in CPU-only mode."
fi

# --- Check for 'pv' (Pipe Viewer) for progress bar ---
if ! command -v pv &> /dev/null; then
    echo "Warning: 'pv' is not installed. Progress bar will not be shown."
    USE_PV=false
else
    USE_PV=true
fi
echo "---"

##################################################################
### WORKLOAD WRAPPER START ###
##################################################################

# This function is called by xargs. 
# It picks a GPU (if available) and runs the command.
run_distributed() {
    local CMD="$1"
    
    # If CPU mode, just run immediately
    if [ "$MODE" = "CPU" ]; then
        eval "$CMD"
        return
    fi

    # -- GPU Load Balancing Logic --
    # Instead of locking (which idles threads), we distribute.
    local DEVICES=($GPU_LIST)
    local NUM_DEVICES=${#DEVICES[@]}
    
    # Simple randomized load balancing
    # We use RANDOM to pick a slot. This statistically fills GPUs evenly 
    # when the number of jobs is large.
    local IDX=$(( RANDOM % NUM_DEVICES ))
    local DEV_ID="${DEVICES[$IDX]}"
            
    # Set context for this specific job
    export PYOPENCL_CTX="$DEV_ID"
    export PYOPENCL_COMPILER_OUTPUT=0
    
    # Run the simulation
    eval "$CMD"
}
export -f run_distributed

##################################################################
### CSV GENERATION FUNCTION ###
##################################################################
generate_csv_for_t() {
    local N_QUBITS_DIR="$1"
    local T_VAL_CURRENT="$2"
    local N_QUBITS_VAL="$3"
    local CSV_FILE="${N_QUBITS_DIR}/summary_t${T_VAL_CURRENT}.csv"

    echo "J,h,z,theta,t,n_qubits,samples" > "$CSV_FILE"

    find "$N_QUBITS_DIR" -name "*_t${T_VAL_CURRENT}.log" | while read -r LOG_PATH; do
        FILENAME=$(basename "$LOG_PATH")
        # Extract params from filename structure
        PARAMS=$(echo "$FILENAME" | sed -e 's/\.log$//' -e 's/J//' -e 's/_h/ /' -e 's/_z/ /' -e 's/_theta/ /' -e 's/_t/ /' -e 's/m/-/g' -e 's/p/./g')
        read -r J_VAL H_VAL Z_VAL THETA_VAL T_VAL <<< "$PARAMS"
        
        # Robust parsing for both result label formats
        RESULT=$(grep -A 1 "## Mean Squared Magnetization ##" "$LOG_PATH" | tail -n 1)
        if [ -z "$RESULT" ] || [[ "$RESULT" == *"##"* ]]; then
             RESULT=$(grep -A 1 "## Output Samples (Decimal Comma-Separated) ##" "$LOG_PATH" | tail -n 1)
        fi
        
        RESULT=$(echo "$RESULT" | xargs)
        if [ -n "$RESULT" ] && [[ "$RESULT" != *"##"* ]]; then
            echo "${J_VAL},${H_VAL},${Z_VAL},${THETA_VAL},${T_VAL},${N_QUBITS_VAL},${RESULT}" >> "$CSV_FILE"
        fi
    done
    printf "CSV summary for T=%s created at %s\n" "$T_VAL_CURRENT" "$CSV_FILE"
}
export -f generate_csv_for_t

# --- Define parameter ranges ---
PI_HALF=$(python3 -c "import numpy as np; print(np.pi/2)")
NEG_PI_HALF=$(python3 -c "import numpy as np; print(-np.pi/2)")
J_VALUES=$(seq -s ' ' -2.0 0.5 2.0)
H_VALUES=$(seq -s ' ' -4.0 0.5 4.0)
Z_VALUES=$(seq -s ' ' -2 1 2)
THETA_VALUES=$(seq -s ' ' $NEG_PI_HALF 0.1 $PI_HALF)
T_VALUES=$(seq -s ' ' 10 1 19)
N_QUBITS_VALUES=$(seq -s ' ' 4 1 2048)

# --- Total iterations calculation ---
NUM_J=$(echo "$J_VALUES" | wc -w)
NUM_H=$(echo "$H_VALUES" | wc -w)
NUM_Z=$(echo "$Z_VALUES" | wc -w)
NUM_THETA=$(echo "$THETA_VALUES" | wc -w)
NUM_T=$(echo "$T_VALUES" | wc -w)
ITERATIONS_PER_QUBIT=$((NUM_J * NUM_H * NUM_Z * NUM_THETA * NUM_T))

echo "Looping through N_QUBITS..."

# Array to hold PIDs of background CSV tasks
csv_pids=()

for N_QUBITS_VAL in $N_QUBITS_VALUES; do
    N_QUBITS_DIR_NAME=$(printf "nqubits_%04d" "$N_QUBITS_VAL")
    N_QUBITS_DIR="${OUTPUT_DIR}/${N_QUBITS_DIR_NAME}"
    mkdir -p "$N_QUBITS_DIR"

    echo "---"
    echo "Starting simulations for n_qubits = ${N_QUBITS_VAL}"

    # Generate command strings
    (
        for J_VAL in $J_VALUES; do
            J_FILE_FORMAT=$(echo "$J_VAL" | sed 's/\./p/g; s/-/m/g')
            for H_VAL in $H_VALUES; do
                H_FILE_FORMAT=$(echo "$H_VAL" | sed 's/\./p/g; s/-/m/g')
                for Z_VAL in $Z_VALUES; do
                    Z_FILE_FORMAT=$(echo "$Z_VAL" | sed 's/\./p/g; s/-/m/g')
                    for THETA_VAL in $THETA_VALUES; do
                        THETA_FILE_FORMAT=$(echo "$THETA_VAL" | sed 's/\./p/g; s/-/m/g')
                        for T_VAL in $T_VALUES; do
                            LOG_FILE="${N_QUBITS_DIR}/J${J_FILE_FORMAT}_h${H_FILE_FORMAT}_z${Z_FILE_FORMAT}_theta${THETA_FILE_FORMAT}_t${T_VAL}.log"
                            if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                                continue
                            fi
                            # Output the RAW python command. The wrapper will handle execution.
                            printf "%s\n" "python3 \"$PYTHON_SCRIPT\" --J \"$J_VAL\" --h \"$H_VAL\" --z \"$Z_VAL\" --theta \"$THETA_VAL\" --t \"$T_VAL\" --n_qubits \"$N_QUBITS_VAL\" > \"$LOG_FILE\" 2>&1"
                        done
                    done
                done
            done
        done
    ) | \
    if [ "$USE_PV" = true ]; then
        # We pipe to xargs which calls our locking wrapper
        pv -l -s "$ITERATIONS_PER_QUBIT" | xargs -P "$PARALLEL_JOBS" -I {} bash -c "run_distributed \"{}\""
    else
        xargs -P "$PARALLEL_JOBS" -I {} bash -c "run_distributed \"{}\""
    fi

    echo "Simulations for n_qubits = ${N_QUBITS_VAL} complete."
    
    echo "Forking CSV summary generation for ${N_QUBITS_DIR_NAME}..."
    (
        printf "%s\n" $T_VALUES | xargs -P "$PARALLEL_JOBS" -I {} bash -c "generate_csv_for_t \"$N_QUBITS_DIR\" \"{}\" \"$N_QUBITS_VAL\""
    ) &
    csv_pids+=($!)
done

echo "---"
echo "Waiting for background CSV jobs..."
for pid in "${csv_pids[@]}"; do
    wait "$pid"
done

echo "---"
echo "All simulations and CSV generations are complete."
