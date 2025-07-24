#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- Helper Functions ---

// Calculates rows and columns for the grid
void factor_width(int width, int* n_rows, int* n_cols) {
    *n_cols = floor(sqrt(width));
    while ((width / *n_cols) * *n_cols != width) {
        (*n_cols)--;
    }
    *n_rows = width / *n_cols;
}

// Calculates nCr (n choose r)
unsigned long long combinations_count(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n / 2) {
        k = n - k;
    }
    unsigned long long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

// Computes closeness-of-like-bits metric C(state)
double closeness_like_bits(int perm, int n_rows, int n_cols) {
    int n_qubits = n_rows * n_cols;
    int like_count = 0;
    int total_edges = 0;

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            int current_pos = i * n_cols + j;
            int s = (perm >> current_pos) & 1;

            // Right neighbor
            int right_pos = i * n_cols + (j + 1) % n_cols;
            int s_right = (perm >> right_pos) & 1;
            like_count += (s == s_right) ? 1 : -1;
            total_edges++;

            // Down neighbor
            int down_pos = ((i + 1) % n_rows) * n_cols + j;
            int s_down = (perm >> down_pos) & 1;
            like_count += (s == s_down) ? 1 : -1;
            total_edges++;
        }
    }
    return (double)like_count / total_edges;
}

// Expected closeness for a given Hamming weight
double expected_closeness_weight(int n_rows, int n_cols, int hamming_weight) {
    int L = n_rows * n_cols;
    if (L < 2) return 0.0;
    unsigned long long same_pairs = combinations_count(hamming_weight, 2) + combinations_count(L - hamming_weight, 2);
    unsigned long long total_pairs = combinations_count(L, 2);
    if (total_pairs == 0) return 0.0;
    double mu_k = (double)same_pairs / total_pairs;
    return 2.0 * mu_k - 1.0;
}

// Generates the next combination of k elements from a set of n
int next_combination(int* combo, int k, int n) {
    int i = k - 1;
    while (i >= 0 && combo[i] == n - k + i) {
        i--;
    }
    if (i < 0) {
        return 0; // No more combinations
    }
    combo[i]++;
    for (int j = i + 1; j < k; j++) {
        combo[j] = combo[j - 1] + 1;
    }
    return 1;
}


// --- Main Logic ---

int main(int argc, char *argv[]) {
    // --- Default settings ---
    int n_qubits = 16;
    int depth = 20;
    int shots = 100;
    double t2 = 1.0;
    double omega = 1.5;
    double J = -1.0;
    double h = 2.0;
    double dt = 0.25;
    double theta = M_PI / 18.0;
    double delta_theta = -1.0; // Flag

    // --- Parse CLI arguments ---
    if (argc > 1) n_qubits = atoi(argv[1]);
    if (argc > 2) depth = atoi(argv[2]);
    if (argc > 3) dt = atof(argv[3]);
    if (argc > 4) shots = atoi(argv[4]);
    if (argc > 5) J = atof(argv[5]);
    if (argc > 6) h = atof(argv[6]);
    if (argc > 7) theta = atof(argv[7]);
    if (argc > 8) delta_theta = atof(argv[8]);

    printf("n_qubits: %d, depth: %d, shots: %d, J: %.2f, h: %.2f\n", n_qubits, depth, shots, J, h);

    omega *= M_PI;
    int n_rows, n_cols;
    factor_width(n_qubits, &n_rows, &n_cols);
    
    if (delta_theta == -1.0) {
        int z = 4;
        double theta_c = asin(h / (z * J));
        delta_theta = theta - theta_c;
    }

    clock_t start = clock();
    
    // --- Bias Calculation ---
    double* bias = (double*)malloc((n_qubits + 1) * sizeof(double));
    double t = depth * dt;

    if (fabs(h) < 1e-9) {
        bias[0] = 1.0;
        for (int i = 1; i <= n_qubits; i++) bias[i] = 0.0;
    } else if (fabs(J) < 1e-9) {
        for (int i = 0; i <= n_qubits; i++) bias[i] = 1.0 / (n_qubits + 1);
    } else {
        double sin_delta_theta = sin(delta_theta);
        double p_decay = (t2 > 0) ? (1.0 + sqrt(t / t2)) : 1.0;
        double p_term_cos = 1.0 + sin_delta_theta * cos(J * omega * t + theta) / p_decay;
        double p = (pow(2.0, fabs(J / h) - 1.0) * p_term_cos - 0.5);
        if (t2 <= 0) p = pow(2.0, fabs(J / h));

        if (p >= 1024) {
            bias[0] = 1.0;
            for (int i = 1; i <= n_qubits; i++) bias[i] = 0.0;
        } else {
            double tot_n = 0;
            for (int q = 0; q <= n_qubits; q++) {
                bias[q] = 1.0 / (n_qubits * pow(2.0, p * q));
                if (isinf(bias[q])) {
                    bias[0] = 1.0;
                    for (int i = 1; i <= n_qubits; i++) bias[i] = 0.0;
                    tot_n = 1.0;
                    break;
                }
                tot_n += bias[q];
            }
            for (int q = 0; q <= n_qubits; q++) {
                bias[q] /= tot_n;
            }
        }
    }

    if (J > 0) { // Antiferromagnetism
        for (int i = 0; i < (n_qubits + 1) / 2; i++) {
            double temp = bias[i];
            bias[i] = bias[n_qubits - i];
            bias[n_qubits - i] = temp;
        }
    }
    
    double* thresholds = (double*)malloc((n_qubits + 1) * sizeof(double));
    double tot_prob = 0;
    for (int q = 0; q <= n_qubits; q++) {
        tot_prob += bias[q];
        thresholds[q] = tot_prob;
    }
    thresholds[n_qubits] = 1.0;

    int* samples = (int*)malloc(shots * sizeof(int));
    srand(time(NULL));

    // --- Sample Generation ---
    for (int s = 0; s < shots; s++) {
        // Step 1: Sample Hamming weight
        double mag_prob = (double)rand() / RAND_MAX;
        int m = 0;
        while (thresholds[m] < mag_prob) {
            m++;
        }

        // Step 2: Sample permutation within Hamming weight
        if (m == 0) {
            samples[s] = 0;
            continue;
        }
        if (m == n_qubits) {
            samples[s] = (1 << n_qubits) - 1;
            continue;
        }

        double closeness_prob_target = (double)rand() / RAND_MAX;
        double current_tot_prob = 0.0;
        
        unsigned long long num_combos = combinations_count(n_qubits, m);
        double expected_closeness = expected_closeness_weight(n_rows, n_cols, m);

        int* combo_indices = (int*)malloc(m * sizeof(int));
        for (int i = 0; i < m; i++) combo_indices[i] = i;

        int final_state_int = 0;
        int combination_found = 0;

        do {
            int state_int = 0;
            for (int i = 0; i < m; i++) {
                state_int |= (1 << combo_indices[i]);
            }

            double closeness = closeness_like_bits(state_int, n_rows, n_cols);
            // The probability of this state, normalized by the number of combinations
            double prob_of_state = ((1.0 + closeness) / (1.0 + expected_closeness)) / num_combos;
            current_tot_prob += prob_of_state;
            
            if (closeness_prob_target <= current_tot_prob) {
                final_state_int = state_int;
                combination_found = 1;
                break;
            }
        } while (next_combination(combo_indices, m, n_qubits));

        if (!combination_found) { // Fallback for floating point inaccuracies
             int state_int = 0;
             for (int i = 0; i < m; i++) state_int |= (1 << combo_indices[i]);
             final_state_int = state_int;
        }

        samples[s] = final_state_int;
        free(combo_indices);
    }

    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Samples: [");
    for(int s = 0; s < shots; s++) {
        printf("%d%s", samples[s], (s == shots - 1) ? "" : ", ");
    }
    printf("]\n");
    printf("Seconds: %f\n", seconds);

    free(bias);
    free(thresholds);
    free(samples);

    return 0;
}
