# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara
# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

# orginal version https://github.com/vm6502q/PyQrackIsing/blob/main/scripts/otoc_validation.py
# mofied by gemini25

import math
import numpy as np
import statistics
import sys

from collections import Counter

# Qiskit Aer, transpile, and circuit-building imports have been removed.
from pyqrackising import generate_otoc_samples


# The factor_width, trotter_step, calc_stats, hamming_distance, 
# and top_n functions have been removed as they were only
# used for the Qiskit Aer validation part.


def main():
    n_qubits = 16
    depth = 16
    t1 = 0
    t2 = 1
    omega = 1.5

    # Quantinuum settings
    J, h, dt, z = -1.0, 2.0, 0.25, 4
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt, z = -1.0, 0.0, 0.25, 4
    # theta = 0

    # Pure transverse field
    # J, h, dt, z = 0.0, 2.0, 0.25, 4
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt, z = -1.0, 1.0, 0.25, 4
    # theta = -math.pi / 4

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        t1 = float(sys.argv[4])
    if len(sys.argv) > 5:
        shots = int(sys.argv[5])
    else:
        shots = max(65536, 1 << (n_qubits + 2))
    if len(sys.argv) > 6:
        trials = int(sys.argv[6])
    else:
        trials = 8 if t1 > 0 else 1

    print("t1: " + str(t1))
    print("t2: " + str(t2))
    print("omega / pi: " + str(omega))

    omega *= math.pi
    
    # The Qiskit circuit construction and AerSimulator
    # control run have been removed.

    shots = 1<<(n_qubits + 2)
    experiment_probs = dict(Counter(generate_otoc_samples(n_qubits=n_qubits, J=J, h=h, z=z, theta=theta, t=dt*depth, shots=shots, pauli_string='X'+'I'*(n_qubits-1), measurement_basis='Z'*n_qubits)))
    experiment_probs = { k: v / shots for k, v in experiment_probs.items() }

    # Removed the calc_stats call, as there is no
    # control/ideal distribution to compare against.
    
    # Print the results from pyqrackising
    print("PyQrackIsing Results:")
    print(experiment_probs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
