// main.c
#define CL_TARGET_OPENCL_VERSION 220
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Helper to check for OpenCL errors
#define CHECK_OCL_ERROR(err, op)                                         \
    if (err != CL_SUCCESS) {                                             \
        fprintf(stderr, "OpenCL Error %d on operation '%s' at %s:%d\n",  \
                err, op, __FILE__, __LINE__);                            \
        exit(1);                                                         \
    }

// --- Utility Functions ---

/**
 * @brief Reads a text file into a string. Used for loading the kernel.
 * This version includes the fix for the -Wunused-result warning.
 */
char *read_file(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = (char *)malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULL; // Failed to allocate memory
    }

    // Capture the return value of fread to ensure the read was successful.
    size_t bytes_read = fread(buffer, 1, size, f);

    // Check if the number of bytes read matches the file size.
    if (bytes_read != size) {
        fprintf(stderr, "Warning: File read error on %s\n", filename);
        free(buffer);
        fclose(f);
        return NULL; // Indicate failure
    }

    fclose(f);
    buffer[size] = '\0'; // Null-terminate the string
    return buffer;
}


// Factors width for 2D lattice
void factor_width(int width, int *rows, int *cols) {
    int col_len = floor(sqrt(width));
    while ((width % col_len) != 0) {
        col_len--;
    }
    *rows = width / col_len;
    *cols = col_len;
}

#ifdef _MSC_VER
#include <intrin.h>
#define popcount __popcnt
#else
#define popcount __builtin_popcount
#endif


// --- Host-side Simulation Logic ---

void calculate_magnetization(cl_float2 *state_vector, int n_qubits, float *avg_mag, float *sq_mag) {
    long num_states = 1L << n_qubits;
    *avg_mag = 0.0f;
    *sq_mag = 0.0f;

    for (long i = 0; i < num_states; ++i) {
        float prob = state_vector[i].x * state_vector[i].x + state_vector[i].y * state_vector[i].y;
        if (prob < 1e-9) continue;

        int ones = popcount(i);
        int zeros = n_qubits - ones;
        float m = (float)(zeros - ones) / (float)n_qubits;

        *avg_mag += prob * m;
        *sq_mag += prob * m * m;
    }
}

int main(int argc, char **argv) {
    // --- Default Parameters ---
    int n_qubits = 16;
    int depth = 20;
    int shots = 32768;
    double t1 = 0.0;
    double t2 = 1.0;
    double omega = 1.5;
    double J = -1.0, h = 2.0, dt = 0.25;
    double theta = M_PI / 18.0;
    double delta_theta = 2.0 * M_PI / 9.0;
    int trials = 1;

    // --- Parse Arguments ---
    if (argc > 1) n_qubits = atoi(argv[1]);
    if (argc > 2) depth = atoi(argv[2]);
    if (argc > 3) dt = atof(argv[3]);
    if (argc > 4) t1 = atof(argv[4]);
    if (argc > 5) shots = atoi(argv[5]);
    if (argc > 6) trials = atoi(argv[6]);

    printf("t1: %f\n", t1);
    printf("t2: %f\n", t2);
    printf("omega / pi: %f\n", omega);
    omega *= M_PI;
    
    int n_rows, n_cols;
    factor_width(n_qubits, &n_rows, &n_cols);
    long num_states = 1L << n_qubits;

    // --- Generate RZZ Pairs ---
    int num_horiz_pairs = n_rows * n_cols;
    int num_vert_pairs = n_rows * n_cols;
    int num_total_pairs = num_horiz_pairs + num_vert_pairs;
    cl_int2 *rzz_pairs = (cl_int2 *)malloc(num_total_pairs * sizeof(cl_int2));
    int pair_count = 0;
    // Horizontal
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            rzz_pairs[pair_count++] = (cl_int2){{r * n_cols + c, r * n_cols + (c + 1) % n_cols}};
        }
    }
    // Vertical
    for (int c = 0; c < n_cols; ++c) {
        for (int r = 0; r < n_rows; ++r) {
            rzz_pairs[pair_count++] = (cl_int2){{r * n_cols + c, ((r + 1) % n_rows) * n_cols + c}};
        }
    }

    // --- OpenCL Setup ---
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Fallback to CPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        CHECK_OCL_ERROR(err, "clGetDeviceIDs");
    }
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_OCL_ERROR(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_OCL_ERROR(err, "clCreateCommandQueue");
    
    char *kernel_source = read_file("ising_kernel.cl");
    if (!kernel_source) {
        fprintf(stderr, "Could not read kernel file 'ising_kernel.cl'\n");
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_OCL_ERROR(err, "clCreateProgramWithSource");
    free(kernel_source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Kernel Build Error:\n%s\n", log);
        free(log);
        CHECK_OCL_ERROR(err, "clBuildProgram");
    }

    cl_kernel init_kernel = clCreateKernel(program, "initialize_state", &err);
    CHECK_OCL_ERROR(err, "clCreateKernel(initialize_state)");
    cl_kernel rzz_kernel = clCreateKernel(program, "apply_all_rzz", &err);
    CHECK_OCL_ERROR(err, "clCreateKernel(apply_all_rzz)");
    cl_kernel rx_kernel = clCreateKernel(program, "apply_rx", &err);
    CHECK_OCL_ERROR(err, "clCreateKernel(apply_rx)");

    // --- Buffers ---
    cl_mem state_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, num_states * sizeof(cl_float2), NULL, &err);
    CHECK_OCL_ERROR(err, "clCreateBuffer(state_buf)");
    cl_mem pairs_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_total_pairs * sizeof(cl_int2), rzz_pairs, &err);
    CHECK_OCL_ERROR(err, "clCreateBuffer(pairs_buf)");
    free(rzz_pairs);

    cl_float2 *host_state_vector = (cl_float2 *)malloc(num_states * sizeof(cl_float2));

    // --- Main Simulation Loop ---
    for (int trial = 0; trial < trials; ++trial) {
        printf("--- Trial %d ---\n", trial + 1);

        // 1. Initialize State
        clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &state_buf);
        clSetKernelArg(init_kernel, 1, sizeof(cl_float), &theta);
        clSetKernelArg(init_kernel, 2, sizeof(cl_int), &n_qubits);
        size_t gws = num_states;
        err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
        CHECK_OCL_ERROR(err, "clEnqueueNDRangeKernel(initialize_state)");
        
        struct timespec start_time, current_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        for (int d = 0; d <= depth; ++d) {
            double t = (double)d * dt;
            double model = (t1 > 0) ? (1.0 - 1.0 / exp(t / t1)) : 1.0;
            
            double d_magnetization = 0;
            double d_sqr_magnetization = 0;
            
            // Analytical Model
            if (fabs(h) < 1e-9) {
                d_magnetization = 1.0;
                d_sqr_magnetization = 1.0;
            } else if (fabs(J) < 1e-9) {
                d_magnetization = 0.0;
                d_sqr_magnetization = 0.0; // More accurately 1/n_qubits, but 0 in thermodynamic limit.
            } else {
                 double p_factor = (t2 > 0)
                    ? (pow(2.0, fabs(J / h) - 1.0) * (1.0 + sin(delta_theta) * cos(J * omega * t + theta) / (1.0 + sqrt(t / t2))) - 0.5)
                    : pow(2.0, fabs(J / h));
                
                if (p_factor >= 1024) {
                    d_magnetization = 1.0;
                    d_sqr_magnetization = 1.0;
                } else {
                    double bias[n_qubits + 1];
                    double tot_n = 0;
                    for (int q = 0; q <= n_qubits; ++q) {
                        bias[q] = 1.0 / (n_qubits * pow(2.0, p_factor * q));
                        tot_n += bias[q];
                    }
                    for (int q = 0; q <= n_qubits; ++q) {
                        bias[q] /= tot_n;
                        double m = (double)(n_qubits - (q << 1)) / n_qubits;
                        d_magnetization += bias[q] * m;
                        d_sqr_magnetization += bias[q] * m * m;
                    }
                }
            }
            if (J > 0) { // Antiferromagnetic
                d_magnetization = -d_magnetization;
            }

            double final_magnetization;
            double final_sqr_magnetization;

            if ((d == 0) || (model < 0.99)) {
                // Run simulation
                if (d > 0) {
                    // --- Trotter Step ---
                    // 1. First half RX
                    float rx_angle = h * dt;
                    float c_rx = cos(rx_angle / 2.0f);
                    float s_rx = sin(rx_angle / 2.0f);
                    clSetKernelArg(rx_kernel, 0, sizeof(cl_mem), &state_buf);
                    clSetKernelArg(rx_kernel, 1, sizeof(cl_float), &c_rx);
                    clSetKernelArg(rx_kernel, 2, sizeof(cl_float), &s_rx);
                    for (int q = 0; q < n_qubits; ++q) {
                        clSetKernelArg(rx_kernel, 3, sizeof(cl_int), &q);
                        clEnqueueNDRangeKernel(queue, rx_kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
                    }
                    
                    // 2. RZZ layer
                    float rzz_angle = 2.0f * J * dt;
                    clSetKernelArg(rzz_kernel, 0, sizeof(cl_mem), &state_buf);
                    clSetKernelArg(rzz_kernel, 1, sizeof(cl_float), &rzz_angle);
                    clSetKernelArg(rzz_kernel, 2, sizeof(cl_mem), &pairs_buf);
                    clSetKernelArg(rzz_kernel, 3, sizeof(cl_int), &num_total_pairs);
                    clEnqueueNDRangeKernel(queue, rzz_kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);

                    // 3. Second half RX
                    for (int q = 0; q < n_qubits; ++q) {
                        clSetKernelArg(rx_kernel, 3, sizeof(cl_int), &q);
                        clEnqueueNDRangeKernel(queue, rx_kernel, 1, NULL, &gws, NULL, 0, NULL, NULL);
                    }
                }
                
                clFinish(queue);
                err = clEnqueueReadBuffer(queue, state_buf, CL_TRUE, 0, num_states * sizeof(cl_float2), host_state_vector, 0, NULL, NULL);
                CHECK_OCL_ERROR(err, "clEnqueueReadBuffer");
                
                float sim_mag, sim_sq_mag;
                calculate_magnetization(host_state_vector, n_qubits, &sim_mag, &sim_sq_mag);

                final_magnetization = model * d_magnetization + (1.0 - model) * sim_mag;
                final_sqr_magnetization = model * d_sqr_magnetization + (1.0 - model) * sim_sq_mag;
            } else {
                final_magnetization = d_magnetization;
                final_sqr_magnetization = d_sqr_magnetization;
            }

            clock_gettime(CLOCK_MONOTONIC, &current_time);
            double seconds = (current_time.tv_sec - start_time.tv_sec) + (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            
            printf("{\"width\": %d, \"depth\": %d, \"trial\": %d, \"magnetization\": %f, \"square_magnetization\": %f, \"seconds\": %f}\n",
                   n_qubits, d, trial + 1, final_magnetization, final_sqr_magnetization, seconds);
        }
    }

    // --- Cleanup ---
    free(host_state_vector);
    clReleaseMemObject(state_buf);
    clReleaseMemObject(pairs_buf);
    clReleaseKernel(init_kernel);
    clReleaseKernel(rzz_kernel);
    clReleaseKernel(rx_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
