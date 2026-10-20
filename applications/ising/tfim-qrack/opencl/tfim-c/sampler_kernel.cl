// sampler_kernel.cl
// OpenCL kernel for the Ising Sampler

// --- Helper Functions for the Kernel ---

// Calculates nCr (n choose r)
// Required for the combination unranking algorithm
unsigned long combinations_count(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n / 2) {
        k = n - k;
    }
    unsigned long res = 1;
    for (int i = 1; i <= k; ++i) {
        // Note: Division and multiplication order is important to avoid large intermediate values
        // and truncation errors.
        if (__builtin_mul_overflow(res, (n - i + 1), &res)) {
            // Handle overflow if necessary, for very large n,k
            // For typical n_qubits < 64, this is less of a concern.
        }
        res /= i;
    }
    return res;
}

// "Unranking" Algorithm: Generates the N-th combination.
// Given a global ID (N), this finds the specific combination of 'k' items from a set of 'n'
// that corresponds to that ID. This allows each thread to work independently.
void get_combination_by_index(unsigned long N, int k, int n, int* result) {
    int current_n = n;
    int current_k = k;
    for (int i = 0; i < k; ++i) {
        while (true) {
            // Combos if we don't include the current largest element
            unsigned long combos_without_current = combinations_count(current_n - 1, current_k - 1);
            if (N < combos_without_current) {
                break; // Found our element
            }
            // Otherwise, skip this element and decrement the rank
            N -= combos_without_current;
            current_n--;
        }
        result[i] = n - current_n; // Store the index of the element found
        current_k--;
        current_n--;
    }
}


// --- Main Kernel ---

__kernel void calculate_closeness_probabilities(
    const int n_qubits,
    const int n_rows,
    const int n_cols,
    const int m, // The target Hamming weight
    const double expected_closeness,
    const unsigned long num_combos,
    __global double* out_probabilities)
{
    // Get the unique ID for this kernel instance (thread)
    unsigned long id = get_global_id(0);

    // If id is out of bounds, do nothing.
    if (id >= num_combos) {
        return;
    }

    // --- Step 1: Generate the unique combination for this thread ---
    // The MAX_QUBITS must match the host code.
    #define MAX_QUBITS 64
    int combo_indices[MAX_QUBITS];
    get_combination_by_index(id, m, n_qubits, combo_indices);

    // Convert the combination of indices into an integer bitmask representation
    int state_int = 0;
    for (int i = 0; i < m; i++) {
        state_int |= (1 << combo_indices[i]);
    }

    // --- Step 2: Calculate closeness for this specific state ---
    int like_count = 0;
    int total_edges = n_qubits * 2; // Each site has 2 edges (right, down) in periodic BC

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            int current_pos = i * n_cols + j;
            int s = (state_int >> current_pos) & 1;

            // Right neighbor
            int right_pos = i * n_cols + (j + 1) % n_cols;
            int s_right = (state_int >> right_pos) & 1;
            like_count += (s == s_right) ? 1 : -1;

            // Down neighbor
            int down_pos = ((i + 1) % n_rows) * n_cols + j;
            int s_down = (state_int >> down_pos) & 1;
            like_count += (s == s_down) ? 1 : -1;
        }
    }
    double closeness = (double)like_count / total_edges;

    // --- Step 3: Calculate the final probability and write it to the output buffer ---
    double prob_of_state = 0.0;
    if ((1.0 + expected_closeness) > 1e-9) {
         prob_of_state = ((1.0 + closeness) / (1.0 + expected_closeness)) / num_combos;
    }
    out_probabilities[id] = prob_of_state;
}
