#!/bin/bash

# This script performs a parameter sweep for the TFIM sample generator.
# WARNING: This will execute 46,431 runs and generate that many files.
# It will take a very long time to complete.

# --- Configuration ---
DEPTH=20
DT=0.25
SHOTS=1024
PYTHON_SCRIPT="generate_tfim_samples.py"
OUTPUT_DIR="tfim_results"

# --- Setup ---
# Create the output directory if it doesn't exist.
mkdir -p "$OUTPUT_DIR"

# Calculate Pi and the required Theta values using the 'bc' calculator.
# Interpreting "one divisional step" as the two endpoints and zero.
PI=$(echo "4*a(1)" | bc -l)
THETA_NEG=$(echo "-1 * $PI / 18" | bc -l)
THETA_POS=$(echo "$PI / 18" | bc -l)
# THETA_VALUES="$THETA_NEG 0 $THETA_POS"
THETA_VALUES="-0.17453292519 0 0.17453292519"

# --- Main Loops ---
echo "Starting parameter sweep. This will take a long time."
echo "Output will be saved in the '${OUTPUT_DIR}' directory."

# Iterate over qubit counts (using perfect squares for the 2D grid).
for WIDTH in 4 9 16 25 36 49 56 64 96 128 256 322; do
  # Iterate over transverse field strength h.
  for H in $(seq 1.0 0.1 2.0); do
    # Iterate over Ising coupling J.
    for J in $(seq -1.0 0.01 1.0); do
      # Iterate over the three theta values.
      for THETA in $THETA_VALUES; do
      
        # Define a unique, descriptive output filename.
        FILENAME="${OUTPUT_DIR}/W${WIDTH}_H${H}_J${J}_T${THETA}.log"

        echo "RUNNING: Width=$WIDTH, h=$H, J=$J, Theta=${THETA}"

        # Execute the python script with the current parameters and
        # redirect its standard output to the log file.
        python3 "$PYTHON_SCRIPT" "$WIDTH" "$DEPTH" "$DT" "$SHOTS" "$J" "$H" "$THETA" > "$FILENAME"
        
      done
    done
  done
done

echo "Parameter sweep finished."
