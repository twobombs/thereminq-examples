# -*- coding: utf-8 -*-
# Import the necessary function from the PyQrackIsing library
# 

from pyqrackising import generate_otoc_samples

# Define the parameters based on the provided image example
J = -1.0                      # Interaction strength parameter
h = 2.0                       # Transverse field strength parameter
z = 4                         # Coordination number (number of neighbors for interactions)
theta = 0.174532925199432957  # Angle parameter (likely related to rotation or evolution)
t = 5                         # Time parameter for the evolution
n_qubits = 56                 # Total number of qubits in the simulation
cycles = 1                    # Number of cycles (perhaps Trotter steps or repetitions)
# Pauli string defining operators: 'X' on the first qubit, 'I' (identity) on the rest.
# This likely defines the initial operator M and/or the butterfly operator B in the OTOC calculation C(t)=Tr[M(t)� B� M(t) B].
pauli_string = 'X' + 'I' * 55
shots = 100                   # Number of measurement shots to simulate
# Measurement basis: Measure all qubits in the Z basis.
measurement_basis = 'Z' * 56

# Call the function to generate OTOC samples
# Note: As stated in the image, this function is experimental and needs systematic validation.
samples = generate_otoc_samples(
    J=J,
    h=h,
    z=z,
    theta=theta,
    t=t,
    n_qubits=n_qubits,
    cycles=cycles,
    pauli_string=pauli_string,
    shots=shots,
    measurement_basis=measurement_basis
)

# Print the resulting samples (the format might vary, could be raw bitstrings or processed results)
print("Generated OTOC Samples:")
print(samples)
root@80ec19f1223f:~# 






