// ising_kernel.cl

/**
 * Complex number multiplication.
 */
inline float2 c_mul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/**
 * @brief Initializes the state vector to the product state |ψ⟩ = ⊗(cos(θ/2)|0⟩ + sin(θ/2)|1⟩).
 * @param state The state vector.
 * @param theta_half The angle θ/2 for the initial RY rotation.
 * @param n_qubits The total number of qubits.
 */
__kernel void initialize_state(__global float2 *state, const float theta_half, const int n_qubits) {
    int gid = get_global_id(0);
    
    float cos_t = cos(theta_half);
    float sin_t = sin(theta_half);
    
    int num_ones = popcount(gid);
    int num_zeros = n_qubits - num_ones;
    
    state[gid] = (float2)(pown(cos_t, num_zeros) * pown(sin_t, num_ones), 0.0f);
}

/**
 * @brief Applies all RZZ gates for the nearest-neighbor interactions.
 * RZZ is a diagonal gate, so all interactions can be applied in a single pass without barriers.
 * @param state The state vector.
 * @param angle The rotation angle (2 * J * dt).
 * @param pairs A buffer of (q1, q2) integer pairs for nearest neighbors.
 * @param num_pairs The total number of interacting pairs.
 */
__kernel void apply_all_rzz(__global float2 *state, const float angle, __global const int2 *pairs, const int num_pairs) {
    int gid = get_global_id(0);
    float total_phase = 0.0f;

    // The phase from RZZ(θ) is exp(-iθ/2) if bits are same, exp(iθ/2) if different.
    // Let's use the angle definition from the python script: RZZGate(2 * J * dt).
    // The angle in RZZGate is the inner parameter, so we apply phase exp(-i*J*dt) or exp(i*J*dt).
    float half_angle = angle / 2.0f;

    for (int i = 0; i < num_pairs; ++i) {
        int q1 = pairs[i].x;
        int q2 = pairs[i].y;
        
        int bit1 = (gid >> q1) & 1;
        int bit2 = (gid >> q2) & 1;
        
        total_phase += (bit1 == bit2) ? -half_angle : half_angle;
    }
    
    float2 phase_factor = (float2)(cos(total_phase), sin(total_phase));
    state[gid] = c_mul(state[gid], phase_factor);
}


/**
 * @brief Applies RX(angle) to a specific qubit `q`.
 * This pattern applies a single gate to the entire state vector in parallel.
 * @param state The state vector.
 * @param c cos(angle/2)
 * @param s sin(angle/2)
 * @param q The target qubit index.
 */
__kernel void apply_rx(__global float2 *state, const float c, const float s, const int q) {
    int gid = get_global_id(0);
    
    // Process each pair of amplitudes only once.
    // The work-item with a 0 at bit `q` handles the update.
    if (((gid >> q) & 1) == 0) {
        int partner_idx = gid | (1 << q);

        float2 amp0 = state[gid];
        float2 amp1 = state[partner_idx];
        
        // RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
        // new_amp0 = c*amp0 - i*s*amp1
        // new_amp1 = c*amp1 - i*s*amp0
        
        float2 i_s_amp1 = (float2)(s * amp1.y, -s * amp1.x); // -i*s*amp1
        float2 i_s_amp0 = (float2)(s * amp0.y, -s * amp0.x); // -i*s*amp0
        
        state[gid].x = c * amp0.x + i_s_amp1.x;
        state[gid].y = c * amp0.y + i_s_amp1.y;

        state[partner_idx].x = c * amp1.x + i_s_amp0.x;
        state[partner_idx].y = c * amp1.y + i_s_amp0.y;
    }
}
