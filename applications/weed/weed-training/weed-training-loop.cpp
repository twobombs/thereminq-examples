// ==========================================
// WEED CONFIGURATION (Must be before includes)
// ==========================================
// FPPOW: 0 = half (FP16), 1 = float, 2 = double
#define FPPOW 0 
// TCAPPOW: 0 = 32-bit, 1 = 64-bit (Required for >4B indices)
#define TCAPPOW 1 

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Core Weed Includes [cite: 577, 638]
#include "weed.hpp" 
#include "tensors/tensor.hpp"
#include "modules/linear.hpp"
#include "autograd/adam.hpp"
#include "autograd/mse_loss.hpp"
#include "autograd/zero_grad.hpp"
#include "storage/sparse_cpu_real_storage.hpp" // For manual sparse hydration

using namespace weed;

// ==========================================
// HELPER: Binary Data Ingestion
// ==========================================
Tensor ingest_sparse_tensor(const std::string& prefix) {
    std::cout << "[Weed] Ingesting " << prefix << "..." << std::endl;

    // 1. Read Metadata
    std::ifstream meta(prefix + ".meta", std::ios::binary);
    if (!meta) throw std::runtime_error("Meta file not found");

    uint64_t rank_val, nnz_val;
    meta.read(reinterpret_cast<char*>(&rank_val), sizeof(uint64_t));
    
    std::vector<tcapint> shape(rank_val);
    meta.read(reinterpret_cast<char*>(shape.data()), rank_val * sizeof(tcapint));
    
    meta.read(reinterpret_cast<char*>(&nnz_val), sizeof(uint64_t));
    tcapint nnz = static_cast<tcapint>(nnz_val);

    // 2. Create Tensor
    // By default on CPU, this initializes SparseCpuRealStorage [cite: 334]
    Tensor T(shape); 

    // 3. Read Binary Data
    std::vector<tcapint> indices(nnz);
    std::vector<real1> values(nnz); // real1 is 'half' due to FPPOW 0

    std::ifstream f_idx(prefix + "_idx.bin", std::ios::binary);
    std::ifstream f_val(prefix + "_val.bin", std::ios::binary);
    
    f_idx.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(tcapint));
    f_val.read(reinterpret_cast<char*>(values.data()), nnz * sizeof(real1));

    // 4. Hydrate Storage (Direct Sparse Write)
    // We access the raw storage to populate the hash map O(1) per item [cite: 86]
    // Note: This relies on T having a method to write raw flat indices, 
    // or we access the storage pointer directly if exposed. 
    // Assuming standard API usage:
    for(tcapint i = 0; i < nnz; ++i) {
        T.set_flat(indices[i], values[i]); 
    }

    std::cout << "[Weed] Loaded Tensor. Shape: " << shape[0] << "x" << shape[1] 
              << " | NNZ: " << nnz << std::endl;
    return T;
}

// ==========================================
// MAIN TRAINING LOOP
// ==========================================
int main(int argc, char** argv) {
    try {
        std::cout << "--- Weed: Minimalist Sovereign AI Infrastructure ---" << std::endl;

        // 1. Load the "Quantum State" Tensor
        // This is your sparse generated data from Python
        Tensor inputs = ingest_sparse_tensor("epoch_0_tensor");
        
        // Autoencoder/Compression task: Target is same as Input
        Tensor targets = inputs.clone(); 

        // 2. Move to Sovereign Hardware (GPU Cluster)
        // Transfers data from CPU Sparse Map to GPU Buffers [cite: 136]
        inputs.gpu(); 
        targets.gpu();

        // 3. Define the "3D Chess" Model
        // Linear Layer: In -> Out. 
        // We use Complex weights to simulate the Hilbert space rotation.
        // DType::COMPLEX ensures first-class complex number arithmetic[cite: 108, 252].
        Linear model(inputs.shape()[1], inputs.shape()[1], DType::COMPLEX);
        
        model.to_gpu(); // Move weights to GPU [cite: 343]

        // 4. Optimizer Setup
        // Using Adam with standard parameters
        Adam optimizer(model.parameters(), 0.001); // lr=1e-3 [cite: 654]

        // 5. Training Epoch
        std::cout << "--- Starting Compression Epoch ---" << std::endl;
        
        // Forward Pass (Functional Equivalence in action)
        // Real Input >> Complex Weights -> Complex Output (Type Promotion) [cite: 154]
        auto output = model.forward(inputs);

        // Loss Calculation
        // Since output is Complex and Target is Real, we cast Target or take Magnitude
        // For "Quantum" compression, we often minimize distance in Complex plane.
        auto loss = mse_loss(output, targets); // [cite: 163]

        std::cout << "Loss: " << loss.item() << std::endl;

        // Backward Pass (Reverse-Mode AD)
        // Triggers the closure execution for gradient calculation [cite: 115, 367]
        loss.backward();

        // Optimizer Step & Zero Gradients
        adam_step(optimizer); // Updates parameters based on .grad [cite: 173]
        zero_grad(model.parameters()); // Resets gradients for next step [cite: 372]

        std::cout << "--- Epoch Complete ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
