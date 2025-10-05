#!/bin/bash

# Determine the number of available CPU cores.
num_cores=$(nproc)

# The outer loop iterates to generate products.
for i in {65535..4294967295..65535}; do
    # Calculate the product.
    prime1=$(matho-primes -c 1 -u "$i")
    prime2=$(matho-primes -c 1 -u "$(($prime1 * 2))")
    product=$(($prime1 * $prime2))

    # This array will hold the PIDs for ALL jobs related to this single product.
    pids=()

    # Loop through each quality value from 2 to 6.
    for quality in {2..6}; do
        # 1. Launch the "non-cpu" job for the current quality.
        # This fulfills the "keep the non cpu loop" requirement.
        python3 factorizing_wrapper.py "$product" "$quality" &
        pids+=($!)

        # 2. Launch "cpu" jobs, now using the SAME quality variable.
        python3 factorizing_wrapper.py "$product" "$quality" "$cpu" &
        pids+=($!)
    done

    # --- Manage the Grand Race ---

    # Wait for the first process from the entire pool to finish.
    wait -n

    # Once a winner is found, terminate all other running processes for this product.
    kill "${pids[@]}" &>/dev/null

    # Clean up any remaining zombie processes.
    wait &>/dev/null
done

echo "Script finished."
