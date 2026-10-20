import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp
import time
import struct
import os

def generate_qml_sparse_tensor(
    n_features: int,
    density: float,
    seed: int | None = None,
    correlation_strength: float = 0.5,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Generates a minimized, FP16 sparse COO tensor for QML density testing.
    
    Logic:
    1. Generates a Watts-Strogatz 'Small World' graph topology.
    2. Filters for upper-triangle to ensure unique edges (preventing double-counting).
    3. Simulates 'entanglement' by diffusing values across neighbors.
    4. Returns a coalesced PyTorch sparse tensor.
    """
    
    # 1. Setup & Reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if n_features > 1000:
        print(f"--- Generating {n_features}x{n_features} QML Tensor (Density: {density:.2%}) ---")

    # 2. Graph Topology (Small World)
    # Ensure k is at least 2 to prevent isolated islands
    k_neighbors = max(2, int(round(n_features * density)))
    
    t0 = time.time()
    # P=0.1 allows for local clustering with some long-range 'entanglement' shortcuts
    G = nx.watts_strogatz_graph(n_features, k_neighbors, p=0.1, seed=seed)
    
    # 3. Adjacency Extraction
    # Use 'to_scipy_sparse_array' (NX 3.0+) 
    adj_coo = nx.to_scipy_sparse_array(G, format='coo')
    
    row = torch.from_numpy(adj_coo.row)
    col = torch.from_numpy(adj_coo.col)
    
    # MASK: Keep only upper-triangle (i < j)
    # This cuts the stored indices by 50%
    mask = row < col
    row, col = row[mask], col[mask]
    
    # 4. Correlation Diffusion (Value Generation)
    # We generate a 'quantum state' vector and diffuse it so connected nodes
    # have similar values.
    
    # Create random signal
    raw_signal = np.random.randn(n_features).astype(np.float32)
    
    # Diffuse signal through the graph
    adj_csr = adj_coo.tocsr()
    neighbor_sums = adj_csr.dot(raw_signal)
    
    # Diffusion equation
    diffused_signal = (1 - correlation_strength) * raw_signal + \
                      (correlation_strength) * (neighbor_sums / k_neighbors)
    
    # Map node values to edge weights (Average of connected nodes)
    edge_weights = (diffused_signal[row] + diffused_signal[col]) / 2.0
    
    # 5. Tensor Construction (FP16 Optimized)
    # Move to GPU *before* creating the sparse object to save CPU RAM if possible
    values = torch.from_numpy(edge_weights).half().to(device)
    indices = torch.stack([row, col]).to(device)
    
    # Create and Coalesce
    sparse_tensor = torch.sparse_coo_tensor(
        indices, 
        values, 
        size=(n_features, n_features),
        dtype=torch.float16,
        device=device
    )
    
    # Explicit coalesce ensures indices are sorted and unique
    sparse_tensor = sparse_tensor.coalesce()
    
    if n_features > 1000:
        print(f"    Gen Time: {time.time()-t0:.2f}s | NNZ: {sparse_tensor._nnz()} | Memory: ~{sparse_tensor._nnz()*6 / 1024**2:.1f} MB")
        
    return sparse_tensor

def export_to_weed_bin(sparse_tensor: torch.Tensor, filename_prefix: str):
    """
    Exports PyTorch sparse tensor to raw binary for C++ Weed ingestion.
    
    Weed Architecture Mapping:
    - Indices: 'tcapint' (Tensor Capacity Integer). We use uint64 for large scale.
    - Values: 'real1'. We use float16 (half) for memory efficiency.
    - Storage: 'std::unordered_map'. Requires flattened 1D indices.
    """
    print(f"\n[Exporter] Serializing {filename_prefix} for C++ Sovereign Infrastructure...")
    t0 = time.time()
    
    # 1. Ensure CPU and correct types
    # We must move to CPU to use numpy file writing
    t_coo = sparse_tensor.coalesce().cpu()
    
    # Weed uses flattened indices: index = row * cols + col
    # We MUST use uint64 to prevent overflow on N=100,000 (100k^2 > 2^32)
    indices = t_coo.indices().numpy().astype(np.uint64)
    values = t_coo.values().numpy().astype(np.float16)
    shape = t_coo.shape
    
    row_indices = indices[0]
    col_indices = indices[1]
    
    # 2. Calculate Flat Indices (Row-Major)
    # This prepares the data for immediate insertion into Weed's SparseCpuRealStorage map
    # formula: flat_index = row * total_columns + col
    total_cols = np.uint64(shape[1])
    flat_indices = row_indices * total_cols + col_indices
    
    # 3. Write Metadata (.meta)
    # Format: [Rank (uint64), Dims (uint64[]), NNZ (uint64)]
    with open(f"{filename_prefix}.meta", "wb") as f:
        rank = len(shape)
        nnz = len(values)
        
        f.write(struct.pack("Q", rank))            # Rank
        f.write(struct.pack(f"{rank}Q", *shape))   # Dimensions
        f.write(struct.pack("Q", nnz))             # NNZ Count

    # 4. Write Data Arrays (.bin)
    # We write flat indices and values separately for fast std::vector::read
    with open(f"{filename_prefix}_idx.bin", "wb") as f:
        f.write(flat_indices.tobytes())
        
    with open(f"{filename_prefix}_val.bin", "wb") as f:
        f.write(values.tobytes())

    # Get file sizes for report
    meta_sz = os.path.getsize(f"{filename_prefix}.meta")
    idx_sz = os.path.getsize(f"{filename_prefix}_idx.bin")
    val_sz = os.path.getsize(f"{filename_prefix}_val.bin")
    total_sz = (meta_sz + idx_sz + val_sz) / (1024**2)

    print(f"[Exporter] Complete ({time.time()-t0:.2f}s).")
    print(f"    - Metadata: {filename_prefix}.meta")
    print(f"    - Indices:  {filename_prefix}_idx.bin (Flat uint64)")
    print(f"    - Values:   {filename_prefix}_val.bin (FP16)")
    print(f"    - Total Footprint: {total_sz:.2f} MB")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Configuration matches your C++ TCAPPOW=1 (64-bit) and FPPOW=0 (Half) settings
    # 100,000 Features with 0.1% density
    N_FEATURES = 100_000 
    DENSITY = 0.001
    
    print(f"--- QML Tensor Generator (Sovereign AI Pipeline) ---")
    
    try:
        # 1. Generate the 'Quantum State' Tensor
        qml_tensor = generate_qml_sparse_tensor(
            n_features=N_FEATURES, 
            density=DENSITY, 
            seed=42, 
            correlation_strength=0.7 # High correlation for entanglement simulation
        )
        
        # Verify stats
        print("\n--- Tensor Stats ---")
        print(f"Shape:        {qml_tensor.shape}")
        print(f"NNZ:          {qml_tensor._nnz()}")
        print(f"Sparsity:     {1.0 - (qml_tensor._nnz() / (N_FEATURES**2)):.6f}")
        
        # 2. Export to Binary for Weed C++
        export_to_weed_bin(qml_tensor, "epoch_0_tensor")
        
        print("\nReady for C++ ingestion. Run the Weed executable now.")
        
    except Exception as e:
        print(f"Fatal Error: {e}")
