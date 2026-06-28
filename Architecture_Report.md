# Architecture Overview

The codebase implements a distributed, multi-boundary quantum simulation engine orchestrating 12 isolated instances of the `QrackSimulator` across a configurable multi-GPU environment. The primary orchestration relies on Python's `multiprocessing` library, assigning each isolated "universe" (patch) to an independent OS process mapped to a specific OpenCL device context.

*   **Process Isolation:** The main orchestrator (`TraversableWormholeEngine`) creates child processes representing 25-qubit boundary patches. Process execution maps strictly to discrete GPU devices to ensure hardware-level isolation.
*   **Inter-Process Communication (IPC):** Control flow and data synchronization are handled exclusively through OS-level duplex pipes (`multiprocessing.Pipe`). The `sync_broadcast` pattern blocks and manages state transitions collectively (e.g., waiting for all workers to compute boundary Z-expectations before routing the unified mean-field kick).
*   **Centralized Mean-Field Routing:** The parent orchestrator functions as the "Unimatrix" (bulk hub), collecting edge observables from all 12 patches via IPC, calculating a global mean field, and broadcasting proportional Hamiltonian kicks back to the local workers.

# PyQrack API & Gate Efficiency

The system is highly tuned for Random Circuit Sampling (RCS) and dense SYK-like entangling operations.

*   **Gate Fallback Mechanisms:** The gate wrappers (`apply_h`, `apply_rx`, `apply_cx`) verify API endpoint availability (e.g., `hasattr(sim, 'cx')`) before defaulting to universal multi-controlled operations (`sim.mcx`) or manual matrix injections (`sim.mtrx`). This ensures optimal hardware execution when native OpenCL kernels exist, while maintaining compatibility.
*   **Initialization Flags:**
    *   `isOpenCL=True`: Enforces execution within the GPU compute context, essential for offloading linear algebra operations.
    *   `isTensorNetwork=False`: Enforces a flat, dense statevector representation. In a highly entangled SYK-RCS environment, tensor networks would quickly reach exponential bond dimensions, making dense statevectors significantly more performant.
    *   `isSchmidtDecompose=False`: Disables internal Schmidt decomposition algorithms. Given the fully connected intra-patch entanglement (K_25 subsets), bipartite entanglement would rapidly saturate, rendering the overhead of calculating Schmidt coefficients computationally detrimental.

# Concurrency & RAM Constraints

Memory bandwidth is a significant consideration given the size of the quantum state vectors being processed.

*   **Probability Dump Memory Footprint:** A single 25-qubit system requires a dense statevector of 2^25 complex amplitudes. Dumping the absolute probabilities requires extracting an array of 2^25 64-bit floats (`np.float64`).
    *   Size per worker: 2^25 * 8 bytes = 268.4 MB.
    *   Unconstrained concurrency for 12 workers dumping simultaneously would require a sudden allocation burst of > 3.2 GB traversing the PCIe bus and filling host RAM, potentially triggering out-of-memory (OOM) killer routines or NUMA domain thrashing on EPYC host processors.
*   **Semaphore Management:** The architecture mitigates RAM thrashing via a strict `ram_semaphore` initialized to 6 slots (`self.ctx.Semaphore(6)`). This caps concurrent `COMPUTE_BENCHMARKS` PCIe transfers, smoothing the host memory pressure by processing probability arrays in controlled batches. It forces early workers to yield to garbage collection (`del probs`, `gc.collect()`) before subsequent workers saturate the system memory.

# OS & Hardware Safeguards

System-level stability relies on strict thread management and deliberate OS-level process hygiene.

*   **Thread Capping:** The inclusion of variables such as `OMP_NUM_THREADS="4"` and `OPENBLAS_NUM_THREADS="4"` prevents core oversubscription. By explicitly throttling the underlying C-extension thread pools, the Python child processes are blocked from spawning uncontrolled threads that could stall the OpenCL GPU command queues.
*   **OpenCL Context Management:** The orchestrator utilizes the `spawn` context (`mp.get_context('spawn')`) rather than the POSIX default `fork`. Forking an existing process that holds OpenCL state frequently leads to driver segmentation faults or corrupted GPU memory pointers. Spawning guarantees a clean OS environment and safe initialization of the OpenCL runtime per worker.
*   **Pipe and Process Hygiene:**
    *   **Child Connection Closure:** The orchestrator explicitly executes `child_conn.close()` in the parent process. Failing to do this leaves an open file descriptor, which blocks `EOFError` propagation if a worker crashes, potentially leaving the parent in a hanging `wait()` loop.
    *   **Zombie Reaping:** The `shutdown` method deliberately executes `p.join(timeout=2)` after `p.terminate()`. This actively reaps the terminated process from the kernel's process table, preventing resource leaks across iterative simulation runs.

# Bottlenecks & Scaling to 30+ Qubits

Scaling the individual patches from 25 to 30 qubits fundamentally alters the hardware requirements.

*   **Memory Exponentiation:** A 30-qubit patch requires 2^30 amplitudes. The probability vector alone consumes ~8.5 GB per worker. Across 12 patches, a full state dump requires > 100 GB of host RAM.
*   **PCIe Bottlenecks:** The current `COMPUTE_BENCHMARKS` method invokes `dump_probabilities()` or `get_state_vector()`, transferring the entire vector across the PCIe bus to host CPU space for XEB/HOG calculation. At 30 qubits, the PCIe bandwidth limitation will heavily stall the orchestrator, making full state extraction computationally unviable.
*   **Scaling Alternatives:** Instead of full statevector dumps to Python memory, the simulation should transition to:
    *   **Batched Sampling:** Utilizing PyQrack's native Monte Carlo sampling (e.g., `measure_shots`) to estimate observables rather than computing exact XEB from the continuous probability distribution.
    *   **Local Subsystem Tracing:** Refactoring the application to utilize tensor network techniques for boundary expectation values rather than forcing global dense operations.

# Recommendations for the Qrack Core

To optimize this specific federated boundary architecture, the following features should be integrated into the native Qrack C++ core to bypass Python-level orchestrator overhead:

1.  **Native OpenCL XEB/HOG Kernels:** The calculation of Cross-Entropy Benchmarking (XEB) and Heavy Output Generation (HOG) requires median calculation and element-wise squaring. Pushing these aggregation kernels directly into OpenCL C++ would completely eliminate the O(2^N) PCIe transfer of the statevector, returning only scalar float values to the Python orchestrator.
2.  **Federated Mean-Field API:** Implement a native `QrackSimulator` network sync operation using MPI or direct NCCL (if NVIDIA based). Allowing multiple isolated `QrackSimulator` contexts to exchange specific marginal expectation values across an interconnect without traversing the Python `multiprocessing.Pipe` layer would vastly accelerate synchronization steps.
3.  **In-Place Batched Amplitude Rescaling:** Currently, gravitational wormhole kicks are applied via localized `apply_rz` gates dependent on an external bulk mean field. Exposing an OpenCL kernel that accepts an array of global kicks for specific boundary subsets would reduce the gate overhead associated with iterating through dictionary payloads in Python.
