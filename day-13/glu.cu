#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <math.h> // for erff, sqrt

// GELU activation kernel
__global__ void gelu_kernel(float* device_data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        device_data[idx] = 0.5f * device_data[idx] * (1.0f + erff(device_data[idx] / sqrtf(2.0f)));
    }
}

int main() {
    const int num_elements = 1000000;
    float* host_array = new float[num_elements];

    // Initialize array with test values
    std::cout << "[INFO] Initializing host array..." << std::endl;
    for (int i = 0; i < num_elements; ++i) {
        host_array[i] = -1.0f * static_cast<float>(i) / 2.0f;
    }

    std::cout << "[INFO] First 10 elements before GELU:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "host_array[" << i << "] = " << host_array[i] << std::endl;
    }

    // Allocate GPU memory
    float* device_array;
    std::cout << "[INFO] Allocating GPU memory..." << std::endl;
    cudaMalloc(&device_array, num_elements * sizeof(float));

    // Copy data to device
    std::cout << "[INFO] Copying data to GPU..." << std::endl;
    cudaMemcpy(device_array, host_array, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);
    std::cout << "[INFO] Launching GELU kernel with " << blocksPerGrid.x << " blocks of " << threadsPerBlock.x << " threads." << std::endl;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_array, num_elements);

    // Sync and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA Kernel failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy back result
    std::cout << "[INFO] Copying data back to host..." << std::endl;
    cudaMemcpy(host_array, device_array, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_array);

    // Print modified array
    std::cout << "[INFO] First 10 elements after GELU:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "host_array[" << i << "] = " << host_array[i] << std::endl;
    }

    delete[] host_array;
    return 0;
}
