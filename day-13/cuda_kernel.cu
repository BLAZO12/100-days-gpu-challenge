#include "cuda_kernels.h"
#include "helper_functions.h"
#include <stdio.h>

__global__ void addBiasKernel(float* linear_output, const float* bias_vector, int batch_size, int output_dim) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = global_index / output_dim;
    int col_index = global_index % output_dim;

    if (row_index < batch_size && col_index < output_dim) {
        int offset = row_index * output_dim + col_index;
        linear_output[offset] += bias_vector[col_index];

        // Debug print: first 10 elements only to avoid flooding
        if (global_index < 10) {
            printf("Bias Add [row=%d, col=%d]: output += bias => %.4f\n",
                   row_index, col_index, linear_output[offset]);
        }
    }
}

void performLinearLayerOperation(
    cublasHandle_t cublas_handle,
    const float* input_matrix,     // [batch_size x input_dim]
    const float* weight_matrix,    // [output_dim x input_dim] (row-major, transposed in sgemm)
    const float* bias_vector,      // [output_dim]
    float* linear_output,          // [batch_size x output_dim]
    int batch_size,
    int input_dim,
    int output_dim
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf("Performing matrix multiplication: \n");
    printf("Input: [%d x %d], Weights: [%d x %d], Output: [%d x %d]\n",
           batch_size, input_dim, output_dim, input_dim, batch_size, output_dim);

    // Perform matrix multiplication: output = input * weights^T
    checkCublasStatus(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,               // Transpose weight_matrix
        CUBLAS_OP_N,               // Don't transpose input_matrix
        output_dim,                // m: output columns
        batch_size,                // n: batch size (rows)
        input_dim,                 // k: input features
        &alpha,
        weight_matrix,             // A^T
        input_dim,
        input_matrix,              // B
        input_dim,
        &beta,
        linear_output,
        output_dim
    ));

    // Launch kernel to add bias
    int total_elements = batch_size * output_dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    printf("Launching bias addition kernel with grid: %d, block: %d\n", grid_size, block_size);

    addBiasKernel<<<grid_size, block_size>>>(linear_output, bias_vector, batch_size, output_dim);

    checkCudaStatus(cudaGetLastError());
    checkCudaStatus(cudaDeviceSynchronize());

    printf("Linear layer operation complete.\n");
}
