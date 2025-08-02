#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNorm(const float* input, float* output, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        extern __shared__ float shared[];
        float* row_data = shared;

        // Load row data into shared memory
        for (int col = threadIdx.y; col < numCols; col += blockDim.y) {
            row_data[col] = input[row * numCols + col];
        }
        __syncthreads();

        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < numCols; col++) {
            mean += row_data[col];
        }
        mean /= numCols;

        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < numCols; col++) {
            float diff = row_data[col] - mean;
            variance += diff * diff;
        }
        variance /= numCols;
        float stddev = sqrtf(variance + 1e-7f);

        // Normalize and store result
        for (int col = threadIdx.y; col < numCols; col += blockDim.y) {
            output[row * numCols + col] = (row_data[col] - mean) / stddev;
        }
    }
}

int main() {
    const int numRows = 10;
    const int numCols = 10;

    float *hostInput, *hostOutput;
    hostInput = (float*)malloc(numRows * numCols * sizeof(float));
    hostOutput = (float*)malloc(numRows * numCols * sizeof(float));

    printf("Initializing input matrix...\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            hostInput[i * numCols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *devInput, *devOutput;
    printf("Allocating device memory...\n");
    cudaMalloc(&devInput, numRows * numCols * sizeof(float));
    cudaMalloc(&devOutput, numRows * numCols * sizeof(float));

    printf("Copying input to device...\n");
    cudaMemcpy(devInput, hostInput, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numRows + blockSize - 1) / blockSize;
    size_t sharedMemorySize = numCols * sizeof(float);

    printf("Launching LayerNorm kernel with gridSize=%d, blockSize=%d, sharedMemorySize=%zu bytes...\n",
           gridSize, blockSize, sharedMemorySize);
    LayerNorm<<<gridSize, blockSize, sharedMemorySize>>>(devInput, devOutput, numRows, numCols);
    cudaDeviceSynchronize();

    printf("Copying result back to host...\n");
    cudaMemcpy(hostOutput, devOutput, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nInput Matrix A (hostInput):\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%.2f ", hostInput[i * numCols + j]);
        }
        printf("\n");
    }

    printf("\nOutput Matrix B (hostOutput - normalized):\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%.2f ", hostOutput[i * numCols + j]);
        }
        printf("\n");
    }

    printf("Freeing memory...\n");
    cudaFree(devInput);
    cudaFree(devOutput);
    free(hostInput);
    free(hostOutput);

    printf("Done.\n");
    return 0;
}
