#!/bin/bash

# === Configuration ===
# Path to the Python script that processes a SINGLE Trotter step
PYTHON_SCRIPT="sqr_magnetisation-iterations_sep-smol.py" 

# Base directory for all output files
LOG_BASE_DIR="tfim_results_final"

# --- Simulation Parameters ---
N_QUBITS_START=4
N_QUBITS_END=960
N_QUBITS_STEP=1

TROTTER_STEPS_START=1
TROTTER_STEPS_END=20

# --- Concurrency Control ---
# Set the maximum number of Python jobs to run at the same time.
# Default to the number of CPU cores if the variable is not already set.
if [ -z "$MAX_CONCURRENT_JOBS" ]; then
    if command -v nproc &> /dev/null; then
        MAX_CONCURRENT_JOBS=$(nproc)
    elif [ -f /proc/cpuinfo ]; then
        MAX_CONCURRENT_JOBS=$(grep -c ^processor /proc/cpuinfo)
    else
        echo "Warning: Could not determine CPU core count. Defaulting to 4 concurrent jobs."
        MAX_CONCURRENT_JOBS=4 # Fallback
    fi
fi

# Ensure the base output directory exists
mkdir -p "$LOG_BASE_DIR"

echo "--- Starting Massively Parallel TFIM Simulation ---"
echo "Python Script: $PYTHON_SCRIPT"
echo "Qubit Range:   $N_QUBITS_START to $N_QUBITS_END (Step: $N_QUBITS_STEP)"
echo "Trotter Range: $TROTTER_STEPS_START to $TROTTER_STEPS_END"
echo "Max Concurrent Jobs: $MAX_CONCURRENT_JOBS"
echo "Output will be saved in: $LOG_BASE_DIR"
echo "---------------------------------------------------------------------"

# This function launches a job and manages concurrency
run_job() {
    # Assign local variables for clarity
    local n_qubits=$1
    local trotter_step=$2

    # Check if the number of running jobs has reached the maximum
    # The '-p' flag to 'jobs' lists the PIDs of background jobs. 'wc -l' counts them.
    while (( $(jobs -p | wc -l) >= MAX_CONCURRENT_JOBS )); do
        # Wait for any single background job to finish before continuing
        wait -n
    done

    echo -e "\rLaunching job: n_qubits=${n_qubits}, t=${trotter_step}           "
    
    # Run the Python script in the background (&)
    # It's crucial that the Python script accepts --trotter_step
    python3 "$PYTHON_SCRIPT" \
        --n_qubits "$n_qubits" \
        --trotter_step "$trotter_step" \
        --log_dir "$LOG_BASE_DIR" &
}

# --- Main Loop ---
# Outer loop for qubits
for (( nq=N_QUBITS_START; nq<=N_QUBITS_END; nq+=N_QUBITS_STEP )); do
    # Inner loop for Trotter steps
    for (( t=TROTTER_STEPS_START; t<=TROTTER_STEPS_END; t++ )); do
        run_job "$nq" "$t"
    done
done

# Wait for all remaining background jobs to complete
echo "All jobs have been launched. Waiting for the final jobs to finish..."
wait

echo "---------------------------------------------------------------------"
echo "All simulations have completed successfully."
echo "Results are located in: $LOG_BASE_DIR"
