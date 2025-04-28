#!/bin/bash

# default for largest device
export QRACK_OCL_DEFAULT_DEVICE=0
# max paging 
export QRACK_MAX_PAGING_QB=29
# lightcone on
export QRACK_QTENSORNETWORK_THRESHOLD_QB=1
# new var for max frambuffer
export QRACK_QBDT_MAX_ALLOC_MB=32000
# Suggested 'WD40' value for SDRP
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.024
# deep fidelity enabled
export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1

cd /notebooks/qrack/pyqrack-examples/ising

for ((i=4; i<=56; i++)); do
  # Array to store PIDs of background jobs for the current 'i'
  pids=()

  echo "--- Starting batch for i=$i ---"
  for ((j=4; j<=i; j++)); do
    echo "Running i=$i, j=$j"
    # Execute python script in the background
    python3 ising_ace.py "$j" "$i" 2>>errors_ising_ace_fixes_$(echo $i)_$(echo $j).txt 1>>ising_aces_$(echo $i)_$(echo $j).txt &
    # Store the PID of the last backgrounded process
    pids+=($!)

  done
  
  echo "--- Waiting for all python scripts launched for i=$i to complete... ---"
  # The 'wait' command waits for the specified PIDs to finish.
  # Using "${pids[@]}" ensures all PIDs in the array are passed correctly.
  wait "${pids[@]}"
  echo "--- All scripts for i=$i completed. Proceeding to next i. ---"

done

echo "--- All batches finished ---"

# end run, do not stop container
tail -f /dev/null
