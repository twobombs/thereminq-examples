#!/bin/bash
# Start with a 32-bit number

i=4294967295

# no lienwrap in BC output
export BC_LINE_LENGTH=0

while true; do

  # Calculate the length in bits
  bits=$(echo "l($i)/l(2)" | bc -l | awk '{print int($0+0.5)}')

  echo
  echo "number of primary bits: $bits"

  # Double the prime number
  bits2=$(($bits + 1))

  # Generate a prime number around the current number of bits
  prime=$(openssl prime -generate -bits $bits)
  prime2=$(openssl prime -generate -bits $bits2)

  # Calculate the product of the two primes
  fact=$(echo "scale=1024; $prime * $prime2" | bc) 

  echo $prime
  echo $prime2
  echo $fact

  # Calculate the length in bits
  factbits=$(echo "l($fact)/l(2)" | bc -l | awk '{print int($0+0.5)}')

  # Print the result
  echo "number of bits in factorial: $factbits"

  # Run qimcifa with the calculated number
  time python3 /FindAFactor/find_a_factor $(echo $fact)

  # Calculate the next power of 2
  i=$(echo "$i * 2" | bc)
done
