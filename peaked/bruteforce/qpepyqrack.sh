#!/bin/bash

# Define Log File
LOGFILE="qpepyqrack.log"

export QRACK_MAX_PAGING_QB=30 QRACK_MAX_CPU_QB=32 


# Clear previous log (overwrite if exists)
echo "Starting Benchmark Sweep (Paging QB First)..." > "$LOGFILE"

# Define Thresholds
THRESHOLDS=(0.1464466 0)

# 1. Outer Loop: Max Paging Qubits (4 to 29)
for PAGING_QB in {33..36}; do
    
    # 2. Inner Loop: Separability Thresholds
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        
        echo "========================================" | tee -a "$LOGFILE"
        echo "RUNNING: PagingQB=$PAGING_QB | Threshold=$THRESHOLD" | tee -a "$LOGFILE"
        
        # --- EXPORTS INSIDE THE LOOP ---
        # Essential to re-export inside the loop so Python picks them up
        export QRACK_OCL_DEFAULT_DEVICE=2
        export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1
        
        # Dynamic variables
        export QRACK_QUNIT_SEPARABILITY_THRESHOLD=$THRESHOLD
	export QRACK_MAX_PAGING_QB=30
        export QRACK_MAX_CPU_QB=$PAGING_QB

        # Run Python script and append both stdout and stderr (time) to log
        { time python3 qpepyqrack.py ; } 2>&1 | tee -a "$LOGFILE"
        
    done
done

echo "Sweep Complete. Check qrack_results.csv for data and $LOGFILE for logs."
