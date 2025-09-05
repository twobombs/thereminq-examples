# -*- coding: utf-8 -*-
import time
import argparse
from PyQrackIsing import tfim_magnetization

# --- Define and parse command-line arguments ---
parser = argparse.ArgumentParser(description="Run a Transverse Field Ising Model (TFIM) simulation.")
parser.add_argument('--J', type=float, default=-1.0, help='Coupling strength.')
parser.add_argument('--h', type=float, default=2.0, help='Transverse field strength.')
parser.add_argument('--z', type=int, default=4, help='Coordination number.')
parser.add_argument('--theta', type=float, default=0.17453292519943295, help='Variational angle in radians (default is pi/18).')
parser.add_argument('--t', type=int, default=20, help='Number of Trotter steps.')
parser.add_argument('--n_qubits', type=int, default=56, help='Number of qubits in the simulation.')

args = parser.parse_args()

# --- Print the parameters ---
print("## Simulation Parameters ##")
print(f"J (Coupling Strength): {args.J}")
print(f"h (Transverse Field): {args.h}")
print(f"z (Coordination Number): {args.z}")
print(f"theta (Variational Angle): {args.theta}")
print(f"t (Trotter Steps): {args.t}")
print(f"n_qubits (Number of Qubits): {args.n_qubits}")
print("-" * 25) # Separator

# --- Time the calculation ---
start_time = time.perf_counter() # Start the clock

samples = tfim_magnetization(
    J=args.J,
    h=args.h,
    z=args.z,
    theta=args.theta,
    t=args.t,
    n_qubits=args.n_qubits,
)

end_time = time.perf_counter() # Stop the clock
elapsed_time = end_time - start_time
print(f"Calculation took: {elapsed_time:.4f} seconds")
print("-" * 25)

# --- Print the output samples as comma-separated decimals ---
print("\n## Output Samples (Decimal Comma-Separated) ##")
print(samples)
