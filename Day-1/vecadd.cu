#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* vec1, const float* vec2, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = vec1[idx] + vec2[idx];
    }
}

int main() {
    const int size = 10;
    float vec1[size], vec2[size], result[size];

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        vec1[i] = i * 1.0f;
        vec2[i] = (size - i) * 1.0f;
    }

    std::cout << "Vector 1: ";
    for (int i = 0; i < size; i++) std::cout << vec1[i] << " ";
    std::cout << "\n";

    std::cout << "Vector 2: ";
    for (int i = 0; i < size; i++) std::cout << vec2[i] << " ";
    std::cout << "\n";

    // Device memory
    float *d_vec1, *d_vec2, *d_result;
    cudaMalloc(&d_vec1, size * sizeof(float));
    cudaMalloc(&d_vec2, size * sizeof(float));
    cudaMalloc(&d_result, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_vec1, vec1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = static_cast<int>(ceil((float)size / blockSize));
    vectorAdd<<<gridSize, blockSize>>>(d_vec1, d_vec2, d_result, size);

    // Copy result back to host
    cudaMemcpy(result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Output result
    std::cout << "Result (vec1 + vec2): ";
    for (int i = 0; i < size; i++) std::cout << result[i] << " ";
    std::cout << "\n";

    // Cleanup
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);

    return 0;
}
