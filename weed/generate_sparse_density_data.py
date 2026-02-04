import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp
import time

def generate_qml_sparse_tensor(
    n_features: int,
    density: float,
    seed: int | None = None,
    correlation_strength: float = 0.5,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Generates a highly optimized, coalesced FP16 Sparse COO Tensor for QML.
    
    Key Features:
    - **Watts-Strogatz Topology:** Mimics quantum circuit connectivity (clusters).
    - **Unique Edges Only:** Filters for upper-triangle to halve memory usage.
    - **Correlated Values:** Diffuses a signal so connected qubits have related values.
    - **Device Safe:** Handles CPU/GPU transfers correctly to avoid indexing errors.
    """
    
    # 1. Setup & Reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Resolve device explicitly
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    if n_features > 1000:
        print(f"--- Generating {n_features:,}x{n_features:,} QML Tensor (Target Density: {density:.2%}) ---")

    t0 = time.time()

    # 2. Graph Generation (CPU)
    # k_neighbors must be at least 2 to ensure connectivity
    k_neighbors = max(2, int(round(n_features * density)))
    G = nx.watts_strogatz_graph(n_features, k_neighbors, p=0.1, seed=seed)

    # 3. Sparse Adjacency & Masking (CPU)
    # Use 'to_scipy_sparse_array' (NetworkX 3.x compliant)
    adj_coo = nx.to_scipy_sparse_array(G, format='coo')
    
    # Convert to standard numpy int64 for processing
    row_np = adj_coo.row
    col_np = adj_coo.col
    
    # MASK: Keep strict upper-triangle (row < col) to remove duplicates
    mask = row_np < col_np
    row_np = row_np[mask]
    col_np = col_np[mask]
    
    # 4. Correlation Diffusion (CPU -> GPU)
    # We calculate the signal on CPU (fast via Scipy) then move to GPU for final assembly
    raw_signal = np.random.randn(n_features).astype(np.float32)
    
    # Efficient sparse matrix-vector multiplication
    # We use the full adjacency (adj_coo) for diffusion so signal spreads in both directions
    neighbor_sums = adj_coo.tocsr().dot(raw_signal)
    
    # Diffuse: Mix self-value with neighbor-average
    # Note: We divide by k_neighbors as an approximation of degree for speed
    diffused_np = (1 - correlation_strength) * raw_signal + \
                  (correlation_strength) * (neighbor_sums / k_neighbors)
    
    # 5. Tensor Assembly (Device Safe)
    # We move data to the target device NOW to avoid "Expected all tensors on same device" errors
    row_tensor = torch.from_numpy(row_np).to(device)
    col_tensor = torch.from_numpy(col_np).to(device)
    diffused_tensor = torch.from_numpy(diffused_np).to(device) # FP32 for now
    
    # Map node values to edge weights on the GPU
    # Edge Weight = Average of the two connected nodes
    edge_weights = (diffused_tensor[row_tensor] + diffused_tensor[col_tensor]) / 2.0
    
    # Cast to FP16 (Half Precision) to save memory
    edge_weights = edge_weights.half()

    # 6. Create & Coalesce
    indices = torch.stack([row_tensor, col_tensor])
    
    sparse_tensor = torch.sparse_coo_tensor(
        indices, 
        edge_weights, 
        size=(n_features, n_features),
        dtype=torch.float16,
        device=device
    )
    
    # Coalesce sorts indices and sets internal flags optimized for sparse matmul
    sparse_tensor = sparse_tensor.coalesce()

    if n_features > 1000:
        nnz = sparse_tensor.nnz() # Public API in newer PyTorch
        # Approx memory: 2 bytes (FP16) + 16 bytes (2x Int64 indices) per element
        mem_mb = (nnz * 18) / (1024**2) 
        print(f"    Gen Time: {time.time()-t0:.2f}s | NNZ: {nnz:,} | Approx GPU Mem: {mem_mb:.2f} MB")
        
    return sparse_tensor

if __name__ == "__main__":
    # --- STRESS TEST CONFIG ---
    N_QUBITS = 100_000   # Large scale
    DENSITY = 0.001      # 0.1% density
    
    try:
        qml_data = generate_qml_sparse_tensor(N_QUBITS, DENSITY, seed=42)
        
        # Validation
        print("\n--- Validation ---")
        print(f"Shape: {qml_data.shape}")
        print(f"Device: {qml_data.device}")
        print(f"Dtype: {qml_data.dtype}")
        
        # Sparse Matrix Multiplication Test (The "Real World" check)
        # Multiply sparse matrix by a random dense vector (FP16)
        print("\n--- Running Sparse MatMul (FP16) ---")
        vec = torch.randn(N_QUBITS, 1, device=qml_data.device, dtype=torch.float16)
        
        t_start = time.time()
        res = torch.sparse.mm(qml_data, vec)
        torch.cuda.synchronize() if qml_data.is_cuda else None
        print(f"MatMul Time: {(time.time() - t_start)*1000:.2f} ms")
        
    except Exception as e:
        print(f"Failed: {e}")
