#define _GNU_SOURCE // Makes M_PI available
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Define the OpenCL target version before including the header
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_QUBITS 64 // Must match kernel definition

// --- Helper Functions ---

void factor_width(int width, int* n_rows, int* n_cols) {
    *n_cols = floor(sqrt(width));
    while ((width / *n_cols) * *n_cols != width) {
        (*n_cols)--;
    }
    *n_rows = width / *n_cols;
}

unsigned long long combinations_count(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    unsigned long long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

double expected_closeness_weight(int n_rows, int n_cols, int hamming_weight) {
    int L = n_rows * n_cols;
    if (L < 2) return 0.0;
    unsigned long long same_pairs = combinations_count(hamming_weight, 2) + combinations_count(L - hamming_weight, 2);
    unsigned long long total_pairs = combinations_count(L, 2);
    if (total_pairs == 0) return 0.0;
    double mu_k = (double)same_pairs / total_pairs;
    return 2.0 * mu_k - 1.0;
}

// --- OpenCL Device Listing Function ---
void list_devices() {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) {
        printf("No OpenCL platforms found.\n");
        return;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    printf("Available OpenCL Devices:\n");
    int device_index = 0;
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);

        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (num_devices == 0) continue;

        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("  [%d] Platform: %s, Device: %s\n", device_index, platform_name, device_name);
            device_index++;
        }
        free(devices);
    }
    free(platforms);
    printf("\nUse the --device <N> flag to select a device.\n");
}


// --- Main Logic ---

int main(int argc, char *argv[]) {
    // --- Parse for device selection flags first ---
    int selected_device_index = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--list-devices") == 0) {
            list_devices();
            return 0;
        }
        if (strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                selected_device_index = atoi(argv[i + 1]);
            }
        }
    }

    if (selected_device_index == -1) {
        list_devices();
        return 0;
    }

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
    double delta_theta = -1.0;

    // --- Parse CLI arguments (excluding device flags) ---
    // Note: A more robust parser would properly skip the --device flags
    if (argc > 1) n_qubits = atoi(argv[1]);
    if (argc > 2) depth = atoi(argv[2]);
    if (argc > 3) dt = atof(argv[3]);
    if (argc > 4) shots = atoi(argv[4]);
    if (argc > 5) J = atof(argv[5]);
    if (argc > 6) h = atof(argv[6]);
    // Skipping other args for simplicity with the new device flag

    printf("n_qubits: %d, depth: %d, shots: %d, J: %.2f, h: %.2f\n", n_qubits, depth, shots, J, h);
    if (n_qubits > MAX_QUBITS) {
        fprintf(stderr, "Error: n_qubits (%d) exceeds MAX_QUBITS (%d).\n", n_qubits, MAX_QUBITS);
        return 1;
    }

    omega *= M_PI;
    int n_rows, n_cols;
    factor_width(n_qubits, &n_rows, &n_cols);

    if (delta_theta == -1.0) {
        int z = 4;
        double theta_c = asin(h / (z * J));
        delta_theta = theta - theta_c;
    }

    // ... (Bias Calculation logic remains identical) ...
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

    if (J > 0) {
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

    // --- OpenCL Setup ---
    cl_device_id device_id = NULL; // The only handle we need here
    cl_int ret;

    // --- Find and select the specified device ---
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    int current_device_index = 0;
    int device_found = 0;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (num_devices == 0) continue;

        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint j = 0; j < num_devices; j++) {
            if (current_device_index == selected_device_index) {
                device_id = devices[j];
                // platform_id is not needed here, so we don't store it.
                device_found = 1;
                break;
            }
            current_device_index++;
        }
        free(devices);
        if (device_found) break;
    }
    free(platforms);

    if (!device_found) {
        fprintf(stderr, "Error: Invalid device index %d.\n", selected_device_index);
        return 1;
    }

    char device_name[128];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    // Load kernel source
    FILE *fp = fopen("sampler_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        return 1;
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    // Create Context and Command Queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    // ... (The rest of the file is identical) ...
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
     if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Kernel Build Error:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "calculate_closeness_probabilities", &ret);

    clock_t start = clock();
    
    for (int s = 0; s < shots; s++) {
        double mag_prob = (double)rand() / RAND_MAX;
        int m = 0;
        while (thresholds[m] < mag_prob) {
            m++;
        }

        if (m == 0) {
            samples[s] = 0;
            continue;
        }
        if (m == n_qubits) {
            samples[s] = (1 << n_qubits) - 1;
            continue;
        }

        unsigned long long num_combos = combinations_count(n_qubits, m);
        if (num_combos == 0) {
             samples[s] = 0;
             continue;
        }

        double expected_closeness = expected_closeness_weight(n_rows, n_cols, m);
        cl_mem out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_combos * sizeof(double), NULL, &ret);
        clSetKernelArg(kernel, 0, sizeof(int), &n_qubits);
        clSetKernelArg(kernel, 1, sizeof(int), &n_rows);
        clSetKernelArg(kernel, 2, sizeof(int), &n_cols);
        clSetKernelArg(kernel, 3, sizeof(int), &m);
        clSetKernelArg(kernel, 4, sizeof(double), &expected_closeness);
        clSetKernelArg(kernel, 5, sizeof(unsigned long long), &num_combos);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &out_buffer);

        size_t global_item_size = num_combos;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
        if(ret != CL_SUCCESS){
            fprintf(stderr, "Failed to enqueue kernel. Error code: %d\n", ret);
            continue;
        }

        double *state_probs = (double*)malloc(num_combos * sizeof(double));
        ret = clEnqueueReadBuffer(command_queue, out_buffer, CL_TRUE, 0, num_combos * sizeof(double), state_probs, 0, NULL, NULL);

        double closeness_prob_target = (double)rand() / RAND_MAX;
        double current_tot_prob = 0.0;
        unsigned long long final_combo_index = num_combos - 1;

        for (unsigned long long i = 0; i < num_combos; i++) {
            current_tot_prob += state_probs[i];
            if (closeness_prob_target <= current_tot_prob) {
                final_combo_index = i;
                break;
            }
        }
        
        int final_state_int = 0;
        int* combo_indices_cpu = (int*)malloc(m * sizeof(int));
        unsigned long long temp_N = final_combo_index;
        int current_n = n_qubits;
        int current_k = m;
        for (int i = 0; i < m; ++i) {
            while (1) {
                unsigned long long combos_without_current = combinations_count(current_n - 1, current_k - 1);
                if (temp_N < combos_without_current) break;
                temp_N -= combos_without_current;
                current_n--;
            }
            combo_indices_cpu[i] = n_qubits - current_n;
            current_k--;
            current_n--;
        }
        for (int i = 0; i < m; i++) {
            final_state_int |= (1 << combo_indices_cpu[i]);
        }
        free(combo_indices_cpu);

        samples[s] = final_state_int;
        free(state_probs);
        clReleaseMemObject(out_buffer);
    }

    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Samples: [");
    for(int s = 0; s < shots; s++) {
        printf("%d%s", samples[s], (s == shots - 1) ? "" : ", ");
    }
    printf("]\n");
    printf("Seconds: %f\n", seconds);

    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(source_str);
    free(bias);
    free(thresholds);
    free(samples);

    return 0;
}
