import time
import numpy as np
import matplotlib.pyplot as plt
from pyqrack import QrackSimulator, Pauli

# --- Configuration ---
n_qubits = 20          # Set to 25 for full scale
J = -1.0
h = 1.0                # Critical Point
total_time = 3.0
n_steps = 100

# Base Filename
base_name = f"tfim_N{n_qubits}_h{h}_t{total_time}_steps{n_steps}"
tensor_file = f"{base_name}_tensor.txt"
plot_file = f"{base_name}_plot.png"

# --- Physics (Native Gates) ---
def apply_step(sim, qubits, J, h, dt):
    # 1. Transverse Field (X)
    for q in qubits:
        sim.r(Pauli.PauliX, 2 * h * dt, q)
    
    # 2. Coupling (ZZ)
    for i in range(len(qubits)):
        q1, q2 = qubits[i], qubits[(i + 1) % len(qubits)]
        sim.mcx([q1], q2)
        sim.r(Pauli.PauliZ, 2 * J * dt, q2)
        sim.mcx([q1], q2)

    # 3. Transverse Field (X) - Second half
    for q in qubits:
        sim.r(Pauli.PauliX, 2 * h * dt, q)

def get_magnetization(sim, qubits):
    m_tot = 0
    for q in qubits:
        m_tot += (1.0 - 2.0 * sim.prob(q))
    return (m_tot / len(qubits))**2

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Initializing Simulation for N={n_qubits}...")
    sim = QrackSimulator(n_qubits)
    qubits = list(range(n_qubits))
    
    # Initialize |+>
    for q in qubits:
        sim.h(q)

    dt = total_time / n_steps
    times = []
    mags = []
    
    # Run Evolution
    start_time = time.time()
    print(f"Running evolution (h={h}, t={total_time})...")
    
    for i in range(n_steps):
        apply_step(sim, qubits, J, h, dt)
        
        # Record Data for Plot
        t_current = (i + 1) * dt
        mag = get_magnetization(sim, qubits)
        times.append(t_current)
        mags.append(mag)
        
        if i % 20 == 0:
            print(f"Step {i}/{n_steps}...")
            
    sim_time = time.time() - start_time
    print(f"Simulation complete in {sim_time:.2f}s.")

    # --- 1. Save Plot ---
    print(f"Saving plot to {plot_file}...")
    plt.figure(figsize=(10, 6))
    plt.plot(times, mags, color='crimson', linewidth=2, label=f'h={h} (Critical)')
    plt.title(f"TFIM Time Evolution\nN={n_qubits}, h={h}", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(r"Order Parameter $\langle Z^2 \rangle$", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(plot_file)
    plt.close() # Free memory

    # --- 2. Save Tensor ---
    print("Extracting State Vector (Tensor)...")
    state_vector = np.array(sim.out_ket())
    
    print(f"Writing tensor to {tensor_file}...")
    write_start = time.time()
    np.savetxt(tensor_file, state_vector, fmt='%.6e%+.6ej', header=f"TFIM State Vector N={n_qubits} h={h} t={total_time}")
    
    print(f"Done. Saved:\n - {plot_file}\n - {tensor_file} ({time.time() - write_start:.2f}s write time)")
