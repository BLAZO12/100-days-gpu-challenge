#include <cuda_runtime.h>
#include <iostream>

// Define matrix dimensions (smaller size for debug prints)
#define MATRIX_WIDTH 1024
#define MATRIX_HEIGHT 1024

// CUDA kernel for matrix transposition
__global__ void transposeMatrix(const float* inputMatrix, float* transposedMatrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int inputIndex = row * width + col;
        int outputIndex = col * height + row;
        transposedMatrix[outputIndex] = inputMatrix[inputIndex];
    }
}

// Utility to check CUDA errors
void checkCudaError(const char* label) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] " << label << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void printMatrix(const float* matrix, int width, int height, const char* name) {
    std::cout << "\n--- " << name << " (First 10x10 section) ---\n";
    for (int i = 0; i < 10 && i < height; i++) {
        for (int j = 0; j < 10 && j < width; j++) {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "-----------------------------\n";
}

int main() {
    int width = MATRIX_WIDTH;
    int height = MATRIX_HEIGHT;
    size_t matrixSize = width * height * sizeof(float);

    // Allocate and initialize host memory
    float* hostInput = (float*)malloc(matrixSize);
    float* hostOutput = (float*)malloc(matrixSize);

    for (int i = 0; i < width * height; i++) {
        hostInput[i] = static_cast<float>(i);
    }

    printMatrix(hostInput, width, height, "Input Matrix");

    // Allocate device memory
    float* deviceInput;
    float* deviceOutput;
    cudaMalloc(&deviceInput, matrixSize);
    cudaMalloc(&deviceOutput, matrixSize);

    // Copy to device
    cudaMemcpy(deviceInput, hostInput, matrixSize, cudaMemcpyHostToDevice);
    checkCudaError("Copy to device failed");

    // Configure execution
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeMatrix<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height);
    cudaDeviceSynchronize();
    checkCudaError("Kernel launch");

    // Copy back result
    cudaMemcpy(hostOutput, deviceOutput, matrixSize, cudaMemcpyDeviceToHost);
    checkCudaError("Copy back to host");

    // Print results
    printMatrix(hostOutput, height, width, "Transposed Matrix");

    // Validate result
    bool correct = true;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (hostOutput[col * height + row] != hostInput[row * width + col]) {
                std::cerr << "[Mismatch] at (" << row << ", " << col << ")\n";
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    std::cout << (correct ? "✅ Matrix transposition succeeded!" : "❌ Matrix transposition failed!") << std::endl;

    // Cleanup
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostInput);
    free(hostOutput);

    return 0;
}
