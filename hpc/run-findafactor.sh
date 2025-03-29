#!/bin/bash

for i in {65535..4294967295..65535}; do
  prime=$(matho-primes -c 1 -u $i)
  i2=$((prime + prime))
  prime2=$(matho-primes -c 1 -u $i2)
  fact=$((prime * prime2))
  python3 /FindAFactor/find_a_factor.py $fact
done
