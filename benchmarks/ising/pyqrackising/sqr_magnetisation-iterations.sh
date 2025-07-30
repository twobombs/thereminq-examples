#!/bin/bash
# Removed debugging output - prints each command as it's executed

# --- Configuration ---
PYTHON_SCRIPT="sqr_magnetisation-iterations_cli.py"
OUTPUT_DIR="tfim_sim_logs"
# TEMP_COMMANDS_FILE="${OUTPUT_DIR}/_commands_to_run.txt" # Removed: No longer needed

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting TFIM batch simulations..."
echo "Output logs will be saved in: $OUTPUT_DIR"
echo "---"

# --- Detect available CPU threads and set parallel jobs ---
# Try to use nproc first, fallback to /proc/cpuinfo if nproc is not available
if command -v nproc &> /dev/null; then
    AVAILABLE_THREADS=$(nproc)
else
    AVAILABLE_THREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)
fi

# Calculate parallel jobs (threads * 4)
PARALLEL_JOBS=$((AVAILABLE_THREADS * 2))

# Ensure at least 1 parallel job
if [ "$PARALLEL_JOBS" -eq 0 ]; then
    PARALLEL_JOBS=1
fi

echo "Detected available CPU threads: $AVAILABLE_THREADS"
echo "Running up to $PARALLEL_JOBS simulations in parallel."
echo "---"

# --- Check for 'pv' (Pipe Viewer) for progress bar ---
if ! command -v pv &> /dev/null; then
    echo "Warning: 'pv' (Pipe Viewer) is not installed. Progress bar will not be shown."
    echo "To install 'pv', use: sudo apt-get install pv (Debian/Ubuntu) or brew install pv (macOS) or equivalent for your OS."
    USE_PV=false
else
    USE_PV=true
fi
echo "---"

# --- Define parameter ranges ---
# J and h: from -2 to 2 with 0.1 steps
J_VALUES=$(seq -s ' ' -2.0 0.1 2.0)
H_VALUES=$(seq -s ' ' -2.0 0.1 2.0)

# Theta: from -4 to 4 with 1 step
THETA_VALUES=$(seq -s ' ' -4 1 4)

# T: from 1 to 20 with 1 step
T_VALUES=$(seq -s ' ' 1 1 20)

# N_qubits: from 4 to 960 with 1 step
N_QUBITS_VALUES=$(seq -s ' ' 4 1 960)

# --- Total iterations warning ---
# Calculate approximate number of iterations for user awareness
NUM_J=$(echo "$J_VALUES" | wc -w)
NUM_H=$(echo "$H_VALUES" | wc -w)
NUM_THETA=$(echo "$THETA_VALUES" | wc -w)
NUM_T=$(echo "$T_VALUES" | wc -w)
NUM_N_QUBITS=$(echo "$N_QUBITS_VALUES" | wc -w)

TOTAL_ITERATIONS=$((NUM_J * NUM_H * NUM_THETA * NUM_T * NUM_N_QUBITS))
echo "Estimated total simulations to run: $TOTAL_ITERATIONS"
echo "This will still take a very long time and generate many files. Proceed with caution."
echo "Press Ctrl+C to cancel at any time."
echo "---"

echo "Executing simulations in parallel..."

# --- Generate and execute commands in parallel using xargs with pv for progress ---
# Commands are now piped directly to xargs, no temporary file is created.
(
    for J_VAL in $J_VALUES; do
        J_FILE_FORMAT=$(echo "$J_VAL" | sed 's/\./p/g' | sed 's/-/m/g')

        for H_VAL in $H_VALUES; do
            H_FILE_FORMAT=$(echo "$H_VAL" | sed 's/\./p/g' | sed 's/-/m/g')

            for THETA_VAL in $THETA_VALUES; do
                THETA_FILE_FORMAT=$(echo "$THETA_VAL" | sed 's/\./p/g' | sed 's/-/m/g')

                for T_VAL in $T_VALUES; do
                    for N_QUBITS_VAL in $N_QUBITS_VALUES; do

                        LOG_FILE="${OUTPUT_DIR}/J${J_FILE_FORMAT}_h${H_FILE_FORMAT}_theta${THETA_FILE_FORMAT}_t${T_VAL}_nqubits${N_QUBITS_VAL}.log"

                        # Echo the full command directly to stdout, which will be piped
                        echo "python \"$PYTHON_SCRIPT\" --J \"$J_VAL\" --h \"$H_VAL\" --theta \"$THETA_VAL\" --t \"$T_VAL\" --n_qubits \"$N_QUBITS_VAL\" > \"$LOG_FILE\" 2>&1"
                    done
                done
            done
        done
    done
) | \
if [ "$USE_PV" = true ]; then
    # Use pv to show progress of lines being piped to xargs
    pv -l -s "$TOTAL_ITERATIONS" | xargs -P "$PARALLEL_JOBS" -I {} bash -c "{}"
else
    # Fallback without pv
    xargs -P "$PARALLEL_JOBS" -I {} bash -c "{}"
fi

# Clean up the temporary commands file (no longer exists, but leaving comment for clarity)
# rm "$TEMP_COMMANDS_FILE"

echo "---"
echo "All simulations complete. Check the '$OUTPUT_DIR' directory for log files."
