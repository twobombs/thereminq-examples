# canary_memory_test.py
import os
import time
import numpy as np

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
os.environ["PYQRACK_SHARED_LIB_PATH"] = QRACK_LIB_PATH
os.environ["DRI_PRIME"] = "pci-0000_44_00_0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OCL_ICD_PLATFORM_SORT"] = "none"
os.environ["RUSTICL_ENABLE"] = "radeonsi"
os.environ["RUSTICL_ALLOW_SVM"] = "0"
os.environ["MESA_VK_DEVICE_SELECT"] = "amd"
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"
os.environ["QRACK_OCL_DEFAULT_DEVICE"] = os.environ.get("WORMHOLE_GPU", "0")
os.environ["QRACK_QPAGER_DEVICES"] = "-1"
os.environ["QRACK_QUNITMULTI_DEVICES"] = "-1"
os.environ["QRACK_FPPOW"] = "5"
# Deliberately high to rely on driver TTM eviction
os.environ["QRACK_MAX_ALLOC_MB"] = "32000" 

from pyqrack import QrackSimulator

def apply_rz(sim, theta, q):
    sim.r(3, float(theta), q)

def apply_zz(sim, theta, q1, q2):
    apply_rz(sim, theta, q1)
    apply_rz(sim, theta, q2)
    ph = complex(np.cos(2.0 * theta), -np.sin(2.0 * theta))
    try:
        sim.mcmtrx([q1], [complex(1, 0), 0j, 0j, ph], q2)
    except TypeError:
        sim.mcmtrx([q1], [complex(1, 0), 0j, 0j, ph], [q2])

def main():
    num_patches = 16
    qubits = 27
    sims = []
    
    print(f"Allocating {num_patches}x {qubits}-qubit simulators (~16GB total)...")
    
    # 1. Allocation & Entanglement (Defeating QUnit factorization)
    for p in range(num_patches):
        tp = time.perf_counter()
        sim = QrackSimulator(qubit_count=qubits)
        for q in range(qubits):
            sim.h(q)
            
        # Apply ZZ ladder to fully entangle the state vector and force actual allocation
        for q in range(qubits - 1):
            apply_zz(sim, 0.1, q, q + 1)
            
        # Force OpenCL queue flush and buffer touch to measure true allocation time
        _ = sim.prob(0)
        sims.append(sim)
        print(f"Patch {p:02d} entangled ({time.perf_counter() - tp:.2f}s).")
        
    # 2. Sequential Execution (Testing TTM eviction)
    print("\nExecuting Round-Robin Trotter Sweep to force paging...")
    
    for pass_n in range(2):
        print(f"\n--- Sweep pass {pass_n} ---")
        t0 = time.perf_counter()
        for p in range(num_patches):
            tp = time.perf_counter()
            sim = sims[p]
            
            # Arbitrary rotation to force state update
            for q in range(qubits):
                sim.r(1, 0.1, q)
                
            # Force OpenCL queue flush to measure true execution + paging time
            _ = sim.prob(0)
            print(f"Swept Patch {p:02d} ({time.perf_counter() - tp:.2f}s)")
            
        print(f"Pass {pass_n} completed in {time.perf_counter() - t0:.2f}s")
        
if __name__ == "__main__":
    main()
