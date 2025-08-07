#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void ELL_kernel(const float* input_matrix, const float* input_vector, float* ell_data,
                           int* ell_indices, float* coo_data, int* coo_rows,
                           int* coo_cols, float* result_vector, const int ell_threshold,
                           const int num_rows, const int num_cols, int* global_coo_counter) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int ell_count = 0;

    for (int col = 0; col < num_cols; ++col) {
        float val = input_matrix[row * num_cols + col];
        if (val != 0) {
            if (ell_count < ell_threshold) {
                ell_data[ell_count * num_rows + row] = val;
                ell_indices[ell_count * num_rows + row] = col;
                ell_count++;

                if (row < 3) printf("ELL[row=%d]: val=%.1f, col=%d\n", row, val, col);
            } else {
                int coo_index = atomicAdd(global_coo_counter, 1);
                coo_data[coo_index] = val;
                coo_rows[coo_index] = row;
                coo_cols[coo_index] = col;

                if (row < 3) printf("COO[row=%d]: val=%.1f, col=%d, index=%d\n", row, val, col, coo_index);
            }
        }
    }

    for (int i = ell_count; i < ell_threshold; ++i) {
        ell_data[i * num_rows + row] = 0;
        ell_indices[i * num_rows + row] = -1;
    }

    float accumulator = 0.0f;
    for (int i = 0; i < ell_threshold; ++i) {
        int col_index = ell_indices[i * num_rows + row];
        if (col_index != -1) {
            accumulator += ell_data[i * num_rows + row] * input_vector[col_index];
        }
    }

    for (int i = 0; i < *global_coo_counter; ++i) {
        if (coo_rows[i] == row) {
            accumulator += coo_data[i] * input_vector[coo_cols[i]];
        }
    }

    result_vector[row] = accumulator;
}

int main() {
    const int num_rows = 1000;
    const int num_cols = 1000;
    const int ell_threshold = 20;

    float* input_matrix = new float[num_rows * num_cols];
    float* ell_data = new float[num_rows * ell_threshold]();
    float* coo_data = new float[num_rows * num_cols]();
    int* ell_indices = new int[num_rows * ell_threshold]();
    int* coo_rows = new int[num_rows * num_cols]();
    int* coo_cols = new int[num_rows * num_cols]();
    float* input_vector = new float[num_cols];
    float* result_vector = new float[num_rows];

    int* d_global_coo_counter;
    CUDA_CHECK(cudaMalloc(&d_global_coo_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_global_coo_counter, 0, sizeof(int)));

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            input_matrix[i * num_cols + j] = (i + j) % 3 == 0 ? i + j : 0;
        }
    }
    for (int i = 0; i < num_cols; i++) {
        input_vector[i] = 1.0f;
    }

    float *d_input_matrix, *d_input_vector, *d_ell_data, *d_coo_data, *d_result_vector;
    int *d_ell_indices, *d_coo_rows, *d_coo_cols;

    CUDA_CHECK(cudaMalloc(&d_input_matrix, num_rows * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_vector, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ell_data, num_rows * ell_threshold * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coo_data, num_rows * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ell_indices, num_rows * ell_threshold * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coo_rows, num_rows * num_cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coo_cols, num_rows * num_cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result_vector, num_rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input_matrix, input_matrix, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_vector, input_vector, num_cols * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    ELL_kernel<<<num_blocks, block_size>>>(
        d_input_matrix, d_input_vector, d_ell_data, d_ell_indices,
        d_coo_data, d_coo_rows, d_coo_cols,
        d_result_vector, ell_threshold, num_rows, num_cols,
        d_global_coo_counter
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds / 1000.0f << " seconds" << std::endl;

    CUDA_CHECK(cudaMemcpy(ell_data, d_ell_data, num_rows * ell_threshold * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(coo_data, d_coo_data, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ell_indices, d_ell_indices, num_rows * ell_threshold * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(coo_rows, d_coo_rows, num_rows * num_cols * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(coo_cols, d_coo_cols, num_rows * num_cols * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_vector, d_result_vector, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    int host_coo_counter;
    CUDA_CHECK(cudaMemcpy(&host_coo_counter, d_global_coo_counter, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Total COO entries: " << host_coo_counter << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << "COO[" << i << "]: val = " << coo_data[i]
                  << ", row = " << coo_rows[i] << ", col = " << coo_cols[i] << std::endl;
    }

    FILE* output_file = fopen("cuda_results.txt", "w");
    if (!output_file) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_rows; i++) {
        fprintf(output_file, "%.10f\n", result_vector[i]);
    }
    fclose(output_file);
    std::cout << "Results written to cuda_results.txt" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_input_matrix));
    CUDA_CHECK(cudaFree(d_input_vector));
    CUDA_CHECK(cudaFree(d_ell_data));
    CUDA_CHECK(cudaFree(d_coo_data));
    CUDA_CHECK(cudaFree(d_ell_indices));
    CUDA_CHECK(cudaFree(d_coo_rows));
    CUDA_CHECK(cudaFree(d_coo_cols));
    CUDA_CHECK(cudaFree(d_result_vector));
    CUDA_CHECK(cudaFree(d_global_coo_counter));

    delete[] input_matrix;
    delete[] ell_data;
    delete[] coo_data;
    delete[] ell_indices;
    delete[] coo_rows;
    delete[] coo_cols;
    delete[] input_vector;
    delete[] result_vector;

    return 0;
}
