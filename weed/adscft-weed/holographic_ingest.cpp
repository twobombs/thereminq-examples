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

const int HILBERT_DIM = 24; 

// ENCODING (2D -> 1D)
size_t encode_boundary_index(const std::vector<float>& row_data) {
    size_t index = 0;
    for (size_t i = 0; i < row_data.size(); i++) {
        int val = static_cast<int>(std::abs(row_data[i]) * 255.0f) % 255;
        index ^= (val << (i * 2)); 
    }
    return index % (1 << HILBERT_DIM);
}

// DECODING (1D -> 2D Visualization)
// This reverses the bit-interleaving to recover approximate X,Y coords
void decode_index(size_t index, int& x, int& y) {
    x = 0; y = 0;
    for (int i = 0; i < 12; i++) { // Assuming 24-bit total, 12 bits per dim
        x |= ((index >> (2 * i)) & 1) << i;
        y |= ((index >> (2 * i + 1)) & 1) << i;
    }
}

// EXPORT FUNCTION
void export_hologram(const std::map<size_t, float>& boundary, const std::string& filename) {
    std::ofstream file(filename);
    file << "index,x,y,amplitude\n"; // CSV Header
    
    int x, y;
    // Standard C++11 loop
    for (std::map<size_t, float>::const_iterator it = boundary.begin(); it != boundary.end(); ++it) {
        decode_index(it->first, x, y);
        file << it->first << "," << x << "," << y << "," << it->second << "\n";
    }
    std::cout << "Hologram exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "--- Phase 1: Holographic Ingestion (Visual Mode) ---" << std::endl;

    // 1. Sparse Accumulator
    std::map<size_t, float> sparse_boundary;

    // 2. Stream Processing
    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Streaming from: " << filename << std::endl;
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
        std::cout << "Simulating 10,000 row stream with drift..." << std::endl;
        for(int i=0; i<10000; ++i) {
            // Create a spiral pattern drift
            float t = i * 0.01f;
            float r = i * 0.00005f;
            std::vector<float> row = {
                0.5f + r * std::cos(t), 
                0.5f + r * std::sin(t)
            };
            size_t idx = encode_boundary_index(row);
            sparse_boundary[idx] += 1.0f;
        }
    }

    std::cout << "Stream Complete. Unique States: " << sparse_boundary.size() << std::endl;

    // --- NEW: EXPORT HOLOGRAM ---
    export_hologram(sparse_boundary, "hologram_vis.csv");

    // 3. Freeze into Tensor (Standard Engine)
    size_t total_size = 1 << HILBERT_DIM;
    std::vector<float> dense_data(total_size, 0.0f);
    
    for (std::map<size_t, float>::iterator it = sparse_boundary.begin(); it != sparse_boundary.end(); ++it) {
        if (it->first < total_size) dense_data[it->first] = it->second;
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
    
    auto bulk_operator = std::make_shared<Tensor>(
        op_data, shape, strides, false, DeviceTag::CPU, 0
    );

    auto result_state = boundary_state * bulk_operator;
    
    // Manual Check
    float overlap = dense_data[target] * op_data[target];
    std::cout << "Hypothesis Overlap at Index " << target << ": " << overlap << std::endl;

    return 0;
}
