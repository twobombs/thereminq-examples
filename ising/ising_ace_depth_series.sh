# advised and tested settings for running the ising models on the ace mitiq backend

export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1
export QRACK_MAX_PAGING_QB=28
export QRACK_QTENSORNETWORK_THRESHOLD_QB=-1
export QRACK_OCL_DEFAULT_DEVICE=2
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466


#!/bin/bash

# This script runs the ising_ace_depth_series.py script
# with the first parameter varying from 4 to 28,
# and the second parameter (depth) fixed at 20.

echo "Starting the script..."

# Loop from 4 to 28 (inclusive)
for i in $(seq 4 46)
do
  echo "Running forked with parameter: $i and depth: $i"

  python3 ising_ace_depth_series.py "$i" "$i" > "$i".log &

  sleep 10

done

echo "Script finished."
