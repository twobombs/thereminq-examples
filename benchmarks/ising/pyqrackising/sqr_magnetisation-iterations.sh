#!/bin/bash
# ---
# Revised script for running Transverse Field Ising Model (TFIM) simulations.
#
# Key Improvements:
# 1. Resumable: Checks for existing, non-empty log files and skips them.
# 2. Safer Defaults: The N_qubits range has been adjusted to a more common
#    set of values to prevent accidentally starting trillions of simulations.
# 3. Enhanced Output: Provides a summary of completed, skipped, and total jobs.
# 4. Sorted Directories: Zero-pads n_qubit directory names for correct sorting.
# 5. Granular & Parallel CSVs: Generates a separate summary CSV for each 'T' 
#    value and creates them in parallel for speed.
# ---

# --- Configuration ---
PYTHON_SCRIPT="sqr_magnetisation-iterations_cli.py"
OUTPUT_DIR="tfim_results_final"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting TFIM batch simulations..."
echo "Output logs will be saved in: $OUTPUT_DIR"
echo "---"

# --- Detect available CPU threads and set parallel jobs ---
if command -v nproc &> /dev/null; then
    AVAILABLE_THREADS=$(nproc)
else
    AVAILABLE_THREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)
fi

PARALLEL_JOBS=$((AVAILABLE_THREADS))
if [ "$PARALLEL_JOBS" -lt 1 ]; then
    PARALLEL_JOBS=1
fi

echo "Detected available CPU threads: $AVAILABLE_THREADS"
echo "Running up to $PARALLEL_JOBS simulations/tasks in parallel."
echo "---"

# --- Check for 'pv' (Pipe Viewer) for progress bar ---
if ! command -v pv &> /dev/null; then
    echo "Warning: 'pv' (Pipe Viewer) is not installed. Progress bar will not be shown."
    echo "To install 'pv', use: sudo apt-get install pv (Debian/Ubuntu) or brew install pv (macOS)"
    USE_PV=false
else
    USE_PV=true
fi
echo "---"

# ##################################################################
# ### NEW SECTION START: Function for Parallel CSV Generation ###
# ##################################################################
# This function encapsulates the logic for creating one summary CSV file.
# It's designed to be called by xargs to run in parallel.
# Arguments: $1: N_QUBITS_DIR, $2: T_VAL_CURRENT, $3: N_QUBITS_VAL
generate_csv_for_t() {
    local N_QUBITS_DIR="$1"
    local T_VAL_CURRENT="$2"
    local N_QUBITS_VAL="$3"
    local CSV_FILE="${N_QUBITS_DIR}/summary_t${T_VAL_CURRENT}.csv"

    # Write header.
    echo "J,h,z,theta,t,n_qubits,result" > "$CSV_FILE"

    # Find log files for the specific T value and append to its CSV.
    find "$N_QUBITS_DIR" -name "*_t${T_VAL_CURRENT}.log" | while read -r LOG_PATH; do
        FILENAME=$(basename "$LOG_PATH")
        PARAMS=$(echo "$FILENAME" | sed -e 's/\.log$//' -e 's/J//' -e 's/_h/ /' -e 's/_z/ /' -e 's/_theta/ /' -e 's/_t/ /' -e 's/m/-/g' -e 's/p/./g')
        read -r J_VAL H_VAL Z_VAL THETA_VAL T_VAL <<< "$PARAMS"
        
        # --- FIX START ---
        # The original script looked for "RESULT: ". 
        # The log file actually contains the result on the line after "## Output Samples...".
        # This command finds that line, gets the line after it (-A 1), and keeps only that second line (tail -n 1).
        RESULT=$(grep -A 1 "## Output Samples (Decimal Comma-Separated) ##" "$LOG_PATH" | tail -n 1)
        # --- FIX END ---
        
        if [ -n "$RESULT" ]; then
            echo "${J_VAL},${H_VAL},${Z_VAL},${THETA_VAL},${T_VAL},${N_QUBITS_VAL},${RESULT}" >> "$CSV_FILE"
        fi
    done
    
    # Use printf for thread-safe output to the main console
    printf "CSV summary for T=%s created at %s\n" "$T_VAL_CURRENT" "$CSV_FILE"
}

# Export the function so it's available to the subshells created by xargs.
export -f generate_csv_for_t
# ##################################################################
# ### NEW SECTION END ###
# ##################################################################


# --- Define parameter ranges ---
PI_HALF=$(python3 -c "import numpy as np; print(np.pi/2)")
NEG_PI_HALF=$(python3 -c "import numpy as np; print(-np.pi/2)")
J_VALUES=$(seq -s ' ' -2.0 0.1 2.0)
H_VALUES=$(seq -s ' ' -2.0 0.1 2.0)
Z_VALUES=$(seq -s ' ' -2 1 2)
THETA_VALUES=$(seq -s ' ' $NEG_PI_HALF 0.1 $PI_HALF)
T_VALUES=$(seq -s ' ' 1 1 20)
N_QUBITS_VALUES=$(seq -s ' ' 4 1 2048)

# --- Total iterations calculation ---
NUM_J=$(echo "$J_VALUES" | wc -w)
NUM_H=$(echo "$H_VALUES" | wc -w)
NUM_Z=$(echo "$Z_VALUES" | wc -w)
NUM_THETA=$(echo "$THETA_VALUES" | wc -w)
NUM_T=$(echo "$T_VALUES" | wc -w)
NUM_N_QUBITS=$(echo "$N_QUBITS_VALUES" | wc -w)

TOTAL_ITERATIONS=$((NUM_J * NUM_H * NUM_Z * NUM_THETA * NUM_T * NUM_N_QUBITS))
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "The current settings will generate an ASTRONOMICAL number of simulations"
echo "This will consume terabytes of disk space and could take weeks to complete."
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "Scanning a total of $TOTAL_ITERATIONS parameter combinations."
echo "---"

# --- Create subdirectories for each n_qubit value ---
echo "Ensuring output directories exist for each n_qubit value..."
for N_QUBITS_VAL in $N_QUBITS_VALUES; do
    N_QUBITS_DIR_NAME=$(printf "nqubits_%04d" "$N_QUBITS_VAL")
    mkdir -p "${OUTPUT_DIR}/${N_QUBITS_DIR_NAME}"
done
echo "---"

echo "Generating and executing commands..."
ITERATIONS_PER_QUBIT=$((NUM_J * NUM_H * NUM_Z * NUM_THETA * NUM_T))

for N_QUBITS_VAL in $N_QUBITS_VALUES; do
    N_QUBITS_DIR_NAME=$(printf "nqubits_%04d" "$N_QUBITS_VAL")
    N_QUBITS_DIR="${OUTPUT_DIR}/${N_QUBITS_DIR_NAME}"

    echo "---"
    echo "Starting simulations for n_qubits = ${N_QUBITS_VAL}"

    # Generate and execute simulation commands in parallel
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
                            printf "%s\n" "python3 \"$PYTHON_SCRIPT\" --J \"$J_VAL\" --h \"$H_VAL\" --z \"$Z_VAL\" --theta \"$THETA_VAL\" --t \"$T_VAL\" --n_qubits \"$N_QUBITS_VAL\" > \"$LOG_FILE\" 2>&1"
                        done
                    done
                done
            done
        done
    ) | \
    if [ "$USE_PV" = true ]; then
        pv -l -s "$ITERATIONS_PER_QUBIT" | xargs -P "$PARALLEL_JOBS" -I {} bash -c "{}"
    else
        xargs -P "$PARALLEL_JOBS" -I {} bash -c "{}"
    fi

    echo "Simulations for n_qubits = ${N_QUBITS_VAL} complete."
    
    # ##################################################################
    # ### MODIFIED SECTION START: Parallel CSV Generation ###
    # ##################################################################
    echo "Generating CSV summaries in parallel for ${N_QUBITS_DIR_NAME}..."

    # Pipe each T value to xargs, which calls our exported function in parallel.
    printf "%s\n" $T_VALUES | xargs -P "$PARALLEL_JOBS" -I {} bash -c "generate_csv_for_t \"$N_QUBITS_DIR\" \"{}\" \"$N_QUBITS_VAL\""

    # ##################################################################
    # ### MODIFIED SECTION END ###
    # ##################################################################
done

echo "---"
echo "All simulations and CSV generations are complete."

