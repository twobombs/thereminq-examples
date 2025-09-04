# `generate_tfim_samples_multi_delta_cpu.sh`

```bash
#!/bin/bash

# This script performs a parameter sweep for the TFIM sample generator.
# It has been optimized to run jobs in parallel to significantly reduce execution time.
# modified for use of C code

# --- Configuration ---
DEPTH=20
DT=0.25
SHOTS=1024
C_SCRIPT="ising_sampler"

# Set this to the number of available GPUs
max_gpu=3

# The output directory is now defined as an absolute path.
# This ensures that output is saved correctly regardless of where the script is run from.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR="${SCRIPT_DIR}/tfim_results"

# Set the number of parallel jobs to run.
# Adjust this based on the number of CPU cores available on your system.
PARALLEL_JOBS=24

# --- Setup ---
# Create the output directory if it doesn't exist.
mkdir -p "$OUTPUT_DIR"

# Define parameter ranges
WIDTH_VALUES="16 25 36 49 56 64 96 128 256 322"
H_VALUES=$(seq 1.0 0.1 2.0)
J_VALUES=$(seq -1.0 0.01 1.0)
THETA_VALUES="-1.57079632679 -0.78539816339 0 0.78539816339 1.57079632679"

# --- Function for a Single Run ---
# This function encapsulates the logic for a single execution of the Python script.
# It's designed to be called by xargs for parallel processing.
run_simulation() {
  # Assign arguments to local variables for clarity
  local width=$1
  local h=$2
  local j=$3
  local theta=$4

  # Define a unique, descriptive output filename using the absolute path.
  local filename="${OUTPUT_DIR}/W${width}_H${h}_J${j}_T${theta}.log"

  echo "STARTING: Width=$width, h=$h, J=$j, Theta=$theta"

  # Execute the c script loadbalanced at a round-robin GPU and redirect its standard output to the log file.
  gpuselect=$((RANDOM % max_gpu))
  ./"$C_SCRIPT" "$width" "$DEPTH" "$DT" "$SHOTS" "$j" "$h" "$theta" > "$filename"

  # Check the exit code of the python script to see if it succeeded.
  if [ $? -eq 0 ]; then
    echo "FINISHED: Width=$width, h=$h, J=$j, Theta=$theta"
  else
    echo "FAILED: Width=$width, h=$h, J=$j, Theta=$theta. See ${filename} for details."
    # You could add logic here to handle failures, like moving the log to an error directory.
    # mkdir -p "${OUTPUT_DIR}/failed"
    # mv "$filename" "${OUTPUT_DIR}/failed/"
  fi
}

# Export the function and variables so they are available to the subshells created by xargs.
export -f run_simulation
export C_SCRIPT max_gpu OUTPUT_DIR DEPTH DT SHOTS

# --- Main Execution ---
# Calculate the total number of runs to provide accurate user feedback.
# Note: The original script's comment of 46,431 runs appears to be incorrect for these ranges.
NUM_WIDTH=$(echo "$WIDTH_VALUES" | wc -w)
NUM_H=$(echo "$H_VALUES" | wc -l)
NUM_J=$(echo "$J_VALUES" | wc -l)
NUM_THETA=$(echo "$THETA_VALUES" | wc -w)
TOTAL_RUNS=$((NUM_WIDTH * NUM_H * NUM_J * NUM_THETA))

echo "=================================================="
echo "Starting parameter sweep with ${TOTAL_RUNS} total runs."
echo "Running up to ${PARALLEL_JOBS} jobs in parallel."
echo "Output will be saved in the '${OUTPUT_DIR}' directory."
echo "This will still take a very long time."
echo "=================================================="

# --- Parameter Generation and Parallel Execution ---
# This block generates all combinations of parameters and pipes them to xargs.
# 'xargs' reads the parameters and executes 'run_simulation' for each set.
# -P "$PARALLEL_JOBS": Sets the maximum number of simultaneous processes.
# -n 4: Tells xargs to use 4 arguments per command invocation.
# bash -c '...': This is the command that xargs runs for each set of arguments.
(
  for width in $WIDTH_VALUES; do
    for h in $H_VALUES; do
      for j in $J_VALUES; do
        for theta in $THETA_VALUES; do
          # Echo the parameters on a single line, separated by spaces.
          # This line is piped as input to xargs.
          echo "$width" "$h" "$j" "$theta"
        done
      done
    done
  done
) | xargs -n 4 -P "$PARALLEL_JOBS" bash -c 'run_simulation "$@"' _

echo "=================================================="
echo "Parameter sweep finished."
echo "=================================================="
```
