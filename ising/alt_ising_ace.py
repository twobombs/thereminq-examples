# Ising model Trotterization simulation using Qiskit and PyQrack

# You likely want to specify environment variable QRACK_MAX_PAGING_QB=N
# where N is >= number of qubits, especially for >28 qubits.
# Example: export QRACK_MAX_PAGING_QB=56

import sys
import time
import os # Imported for environment variable check

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
# from qiskit.providers.basic_provider import BasicSimulator # For potential fallback or comparison

# PyQrack import
try:
    from pyqrack import QrackSimulator
except ImportError:
    print("Error: pyqrack not found.")
    print("Please install it, e.g., using 'pip install pyqrack'")
    sys.exit(1)

def trotter_step(circ: QuantumCircuit, qubits: list, lattice_shape: tuple, J: float, h: float, dt: float):
    """Applies one symmetric Trotter step for the 2D Ising model.

    Args:
        circ: The Qiskit QuantumCircuit to add gates to.
        qubits: A list of qubit indices (assumed to be mapped 0 to N-1).
        lattice_shape: A tuple (n_rows, n_cols).
        J: The ZZ coupling strength.
        h: The transverse field strength.
        dt: The time step duration for this Trotter step.

    Returns:
        The QuantumCircuit with the Trotter step added.

    Notes on Hamiltonian and Gate Implementation:
        - Assumes H = H_ZZ + H_X
        - H_X = -h * sum(X_i)
        - H_ZZ = +J * sum(<ij>) Z_i Z_j  <-- Note the sign of J used here matches the RZZ implementation below.
          If the intended Hamiltonian is H_ZZ = -J * sum(<ij>) Z_i Z_j, the rzz_angle should be -2 * J * dt.

        - Transverse field term: exp(-i * H_X * dt / 2) = exp(-i * (-h * sum(X_i)) * dt / 2) = exp(i * h * dt/2 * sum(X_i))
          Implemented using circ.rx(-h * dt, q) for each qubit, since RX(theta) = exp(-i * theta * X / 2).
          We need -theta/2 = h*dt/2 => theta = -h*dt.

        - ZZ interaction term: exp(-i * H_ZZ * dt) = exp(-i * (+J * sum(<ij>) Z_i Z_j) * dt) = exp(-i * J * dt * sum(<ij>) Z_i Z_j)
          Implemented using RZZ(2 * J * dt) for each pair <ij>, since RZZ(theta) = exp(-i * theta * Z_i Z_j / 2).
          We need -theta/2 = -J*dt => theta = 2 * J * dt.
    """
    n_rows, n_cols = lattice_shape
    num_qubits = len(qubits) # Should match n_rows * n_cols

    # Qubit mapping: Assume qubit index q corresponds to row = q // n_cols, col = q % n_cols

    # --- First half of transverse field term ---
    # Apply Rx(-h * dt) to each qubit
    rx_angle = -h * dt
    for q_index in range(num_qubits):
        circ.rx(rx_angle, qubits[q_index])

    circ.barrier(label="RZZ") # Barrier before ZZ interactions

    # --- ZZ interaction term (using RZZ gates) ---
    # These RZZ gates will be decomposed by the transpiler later if 'rzz'
    # is not in the final basis_gates list.
    rzz_angle = 2 * J * dt

    # Helper to add RZZ gates for a list of pairs
    def add_rzz_pairs(pairs):
        for q_idx1, q_idx2 in pairs:
            # Ensure indices are within the circuit bounds (using qubits list)
            if 0 <= q_idx1 < len(qubits) and 0 <= q_idx2 < len(qubits):
                 circ.append(RZZGate(rzz_angle), [qubits[q_idx1], qubits[q_idx2]])
            else:
                print(f"Warning: Invalid qubit index generated ({q_idx1} or {q_idx2}) - skipping pair.")


    # Apply RZZ gates in layers to cover nearest neighbors on the 2D grid
    # Using open boundary conditions as implied by the loops

    # Layer 1 & 2: Horizontal pairs
    horiz_pairs_layer1 = []
    horiz_pairs_layer2 = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            q_idx1 = r * n_cols + c
            q_idx2 = r * n_cols + (c + 1)
            if c % 2 == 0: # Even columns in layer 1
                 horiz_pairs_layer1.append((q_idx1, q_idx2))
            else:          # Odd columns in layer 2
                 horiz_pairs_layer2.append((q_idx1, q_idx2))

    if horiz_pairs_layer1: add_rzz_pairs(horiz_pairs_layer1)
    if horiz_pairs_layer2: add_rzz_pairs(horiz_pairs_layer2)

    # Layer 3 & 4: Vertical pairs
    vert_pairs_layer1 = []
    vert_pairs_layer2 = []
    for c in range(n_cols):
        for r in range(n_rows - 1):
            q_idx1 = r * n_cols + c
            q_idx2 = (r + 1) * n_cols + c
            if r % 2 == 0: # Even rows in layer 1
                vert_pairs_layer1.append((q_idx1, q_idx2))
            else:          # Odd rows in layer 2
                vert_pairs_layer2.append((q_idx1, q_idx2))

    if vert_pairs_layer1: add_rzz_pairs(vert_pairs_layer1)
    if vert_pairs_layer2: add_rzz_pairs(vert_pairs_layer2)

    circ.barrier(label="RX") # Barrier before second Rx set

    # --- Second half of transverse field term ---
    for q_index in range(num_qubits):
        circ.rx(rx_angle, qubits[q_index])

    return circ

def main():
    # --- Configuration ---
    depth = 1 # Default number of Trotter steps
    if len(sys.argv) > 1:
        try:
            depth = int(sys.argv[1])
            if depth <= 0:
                 raise ValueError("Depth must be positive.")
        except ValueError as e:
            print(f"Error: Invalid command-line argument for Trotter steps depth. {e}")
            print("Usage: python script_name.py [depth]")
            return 1 # Indicate error

    # Lattice dimensions
    n_rows, n_cols = 8, 7
    n_qubits = n_rows * n_cols

    # Ising model parameters
    J = 1.0 # ZZ coupling strength (Ensure sign matches RZZ implementation in trotter_step)
    h = 1.0 # Transverse field strength
    dt = 0.1 # Trotter time step
    total_time = depth * dt

    print(f"--- Ising Model Simulation ---")
    print(f"Grid: {n_rows}x{n_cols} ({n_qubits} qubits)")
    print(f"Parameters: J={J}, h={h}, dt={dt}")
    print(f"Trotter steps (depth): {depth} (Total time: {total_time:.2f})")
    print(f"Using QrackSimulator from pyqrack.")

    # Check for suggested environment variable
    if n_qubits > 28 and os.environ.get('QRACK_MAX_PAGING_QB') is None:
         print("\nWarning: Consider setting environment variable QRACK_MAX_PAGING_QB")
         print(f"         (e.g., export QRACK_MAX_PAGING_QB={n_qubits} or higher) for potentially >28 qubit simulations.\n")


    # --- Circuit Construction ---
    qubit_list = list(range(n_qubits))
    qc = QuantumCircuit(n_qubits, name="IsingTrotter")

    # Build the Trotter circuit by applying steps sequentially
    print("Building Trotter circuit...")
    build_start_time = time.perf_counter()
    for i in range(depth):
        trotter_step(qc, qubit_list, (n_rows, n_cols), J, h, dt)
        # Add barrier between steps for visual clarity (optional)
        if i < depth - 1:
             qc.barrier(label=f"Step {i+1}")
    build_end_time = time.perf_counter()
    print(f"Circuit construction time: {build_end_time - build_start_time:.4f} seconds.")
    print(f"Initial circuit size (operations): {qc.size()}")


    # --- Transpilation ---
    # Define basis gates suitable for QrackSimulator.
    # *** FIX APPLIED HERE (Revision 3) ***
    # Removed "rzz" because QrackSimulator didn't recognize the 'rzz' instruction name.
    # This forces the transpiler to decompose RZZ gates into simpler gates (like cx, rz)
    # which QrackSimulator should understand.
    basis_gates = ["u", "cx", "cy", "cz", "cp", "csx", "id", "rx", "rz", "swap", "iswap"]
    # Note: Depending on the Qrack version and its Qiskit compatibility layer,
    # even 'u' might sometimes need to be decomposed further. If errors persist,
    # a minimal basis like ['id', 'rz', 'rx', 'cx', 'swap'] could be tried.

    print(f"\nTranspiling circuit to basis gates (forcing RZZ decomposition): {basis_gates}...")

    try:
        transpile_start_time = time.perf_counter()
        # Use optimization_level=1 for light optimization; adjust as needed (0 to 3)
        # The transpiler will now break down the original RZZ gates.
        qc_transpiled = transpile(qc, basis_gates=basis_gates, optimization_level=1)
        transpile_end_time = time.perf_counter()
        print(f"Transpilation finished in {transpile_end_time - transpile_start_time:.4f} seconds.")
        # Expect the transpiled circuit size to be larger now due to RZZ decomposition
        print(f"Transpiled circuit size (operations): {qc_transpiled.size()}")

    except ValueError as e:
        print(f"\nError during transpilation: {e}")
        print("This might be due to Qiskit version changes or the chosen basis gates.")
        print("If the error persists, consider using the 'target=' argument for transpile or simplifying the basis_gates list further.")
        return 1 # Indicate error
    except Exception as e:
        print(f"\nAn unexpected error occurred during transpilation: {e}")
        return 1


    # --- Simulation ---
    # Initialize the Qrack simulator
    # Options: isTensorNetwork=True/False, isSchmidtDecomposeMulti=True/False, etc.
    sim = QrackSimulator(n_qubits, isTensorNetwork=False)
    print(f"\nInitialized QrackSimulator for {n_qubits} qubits.")
    # Qrack might print device info here, like in the user's traceback example.

    # Run the simulation (statevector)
    print("Running simulation...")
    sim_start_time = time.perf_counter()
    # Use the transpiled circuit
    sim.run_qiskit_circuit(qc_transpiled, shots=0) # shots=0 -> statevector simulation
    sim_end_time = time.perf_counter()
    print(f"Simulation completed in {sim_end_time - sim_start_time:.4f} seconds.")


    # --- Results ---
    print(f"\n--- Simulation Results ---")
    try:
        # Note: get_unitary_fidelity() measures the simulator's internal numerical precision
        # against its own ideal gate applications, *not* the Trotter approximation error.
        # It might be 1.0 unless numerical issues occur within Qrack.
        sim_fidelity = sim.get_unitary_fidelity()
        print(f"Simulator numerical fidelity: {sim_fidelity:.8f}")
    except Exception as e:
        print(f"Could not get simulator fidelity: {e}")

    # Perform a measurement (collapses the statevector)
    # m_all() returns an integer representing the measured bitstring
    measurement_result_int = sim.m_all()

    # Format measurement result as a binary string
    measurement_result_bin = bin(measurement_result_int)[2:].zfill(n_qubits)
    print(f"Final measurement outcome (binary): {measurement_result_bin}")
    # print(f"Final measurement outcome (integer): {measurement_result_int}") # Optional

    print("\nSimulation finished successfully.")
    return 0 # Indicate success


if __name__ == '__main__':
    # Get the current date and time
    # Using UTC for consistency, matching user's traceback time zone
    now_utc = time.gmtime()
    print(f"Execution started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', now_utc)}")

    # Run the main function and exit with its return code
    exit_code = main()

    now_utc = time.gmtime()
    print(f"Execution finished: {time.strftime('%Y-%m-%d %H:%M:%S UTC', now_utc)}")
    sys.exit(exit_code)
