#!/bin/bash

# The outer loop iterates from 65535 to 4294967295 with a step of 65535
for i in {65535..4294967295..65535}; do
    # Find the largest prime number less than or equal to $i
    prime1=$(matho-primes -c 1 -u "$i")

    # Find the largest prime less than or equal to (prime1 * 2)
    prime2=$(matho-primes -c 1 -u "$(($prime1 * 2))")

    # Calculate the product of the two primes
    product=$(($prime1 * $prime2))

    # --- Start of concurrent execution block ---

    # Array to hold the PIDs of the background jobs for this product
    pids=()

    # Launch all quality iterations concurrently
    for quality in {2..6}; do
        python3 factorizing_wrapper.py "$product" "$quality" &
        # Store the PID of the last backgrounded process in our array
        pids+=($!)
    done

    # Wait for the *first* of the background jobs to finish.
    # The -n flag is the key here.
    wait -n

    # As soon as one process finishes, kill all the PIDs we stored.
    # This terminates the other four running processes.
    # Output is redirected to /dev/null to suppress messages.
    kill "${pids[@]}" &>/dev/null

    # A final wait cleans up any remaining terminated (zombie) processes.
    wait &>/dev/null

    # --- End of concurrent execution block ---
done

echo "Script finished."
