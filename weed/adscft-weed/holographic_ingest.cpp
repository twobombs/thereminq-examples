#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <map> 

// --- LOCAL INCLUDES ---
#include "shared_api.hpp" 
#include "tensors/tensor.hpp"
#include "enums/device_tag.hpp" 
#include "ops/commuting.hpp" 

using namespace Weed; 

// CONFIGURATION
const int HILBERT_DIM = 24; 

// HOLOGRAPHIC ENCODING
size_t encode_boundary_index(const std::vector<float>& row_data) {
    size_t index = 0;
    for (size_t i = 0; i < row_data.size(); i++) {
        int val = static_cast<int>(std::abs(row_data[i]) * 255.0f) % 255;
        index ^= (val << (i * 2)); 
    }
    return index % (1 << HILBERT_DIM);
}

int main(int argc, char* argv[]) {
    std::cout << "--- Phase 1: Holographic Ingestion (Streaming Mode) ---" << std::endl;

    // 1. Sparse Accumulator
    std::map<size_t, float> sparse_boundary;

    // 2. Stream Processing
    if (argc > 1) {
        // REAL FILE MODE
        std::string filename = argv[1];
        std::cout << "Streaming from file: " << filename << std::endl;
        std::ifstream file(filename);
        std::string line;
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<float> features;
            
            while (std::getline(ss, cell, ',')) {
                try { features.push_back(std::stof(cell)); } catch (...) {}
            }
            
            if (!features.empty()) {
                size_t idx = encode_boundary_index(features);
                sparse_boundary[idx] += 1.0f;
            }
        }
    } else {
        // DEMO STREAM MODE
        std::cout << "No file provided. Simulating 10,000 row stream..." << std::endl;
        for(int i=0; i<10000; ++i) {
            std::vector<float> row = {0.5f + (i * 0.0001f), 0.1f + (i * 0.0001f)};
            size_t idx = encode_boundary_index(row);
            sparse_boundary[idx] += 1.0f;
        }
    }

    std::cout << "Stream Complete." << std::endl;
    std::cout << "Unique Holographic States: " << sparse_boundary.size() << std::endl;

    // 3. Freeze into Weed Tensor
    std::cout << "Freezing into Sparse Tensor..." << std::endl;
    
    size_t total_size = 1 << HILBERT_DIM;
    std::vector<float> dense_data(total_size, 0.0f);
    
    // FIX: Replaced C++17 structured binding with C++11 syntax
    for (auto const& pair : sparse_boundary) {
        size_t key = pair.first;
        float val = pair.second;
        
        if (key < total_size) {
            dense_data[key] = val;
        }
    }

    std::vector<unsigned int> shape = {static_cast<unsigned int>(total_size)};
    std::vector<unsigned int> strides = {1}; 
    
    auto boundary_state = std::make_shared<Tensor>(
        dense_data, shape, strides, false, DeviceTag::CPU, 0
    );

    // --- Phase 2: Inference ---
    std::cout << "\n--- Phase 2: Bulk Reconstruction ---" << std::endl;
    
    size_t target = 27; 
    
    std::vector<float> op_data(total_size, 0.0f);
    op_data[target] = 1.0f; 
    if(target > 0) op_data[target-1] = 0.5f;
    if(target < total_size-1) op_data[target+1] = 0.5f;

    auto bulk_operator = std::make_shared<Tensor>(
        op_data, shape, strides, false, DeviceTag::CPU, 0
    );

    auto result_state = boundary_state * bulk_operator;
    
    // Manual Check
    float overlap = dense_data[target] * op_data[target];
    std::cout << "Hypothesis Overlap at Index " << target << ": " << overlap << std::endl;

    return 0;
}
