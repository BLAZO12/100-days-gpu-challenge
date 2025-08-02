#define LOAD_SIZE 32
#include <iostream>
#include <cuda_runtime.h>

// Brent-Kung Parallel Prefix Sum (Scan)
__global__ void prefixsum_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x;
    int globalIdx = 2 * blockDim.x * blockIdx.x + tid;

    __shared__ float sharedData[LOAD_SIZE];

    // Load data into shared memory
    if (globalIdx < N)
        sharedData[tid] = input[globalIdx];
    else
        sharedData[tid] = 0.0f;

    if (globalIdx + blockDim.x < N)
        sharedData[tid + blockDim.x] = input[globalIdx + blockDim.x];
    else
        sharedData[tid + blockDim.x] = 0.0f;

    __syncthreads();

    // Up-Sweep (Reduce) Phase
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index < LOAD_SIZE) {
            sharedData[index] += sharedData[index - stride];
            printf("Up-Sweep: stride=%d, index=%d, val=%.2f\n", stride, index, sharedData[index]);
        }
    }

    __syncthreads();

    // Down-Sweep Phase
    for (int stride = LOAD_SIZE / 4; stride >= 1; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < LOAD_SIZE) {
            sharedData[index + stride] += sharedData[index];
            printf("Down-Sweep: stride=%d, index=%d, val=%.2f\n", stride, index + stride, sharedData[index + stride]);
        }
    }

    __syncthreads();

    // Write results to global memory
    if (globalIdx < N)
        output[globalIdx] = sharedData[tid];
    if (globalIdx + blockDim.x < N)
        output[globalIdx + blockDim.x] = sharedData[tid + blockDim.x];

    __syncthreads();
}


// Error checking helper
void checkCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error (%s): %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main() {
    int N = 10;
    float hostInput[N], hostOutput[N];

    for (int i = 0; i < N; i++) {
        hostInput[i] = i + 1.0f;
    }

    float *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, N * sizeof(float));
    cudaMalloc(&deviceOutput, N * sizeof(float));

    cudaMemcpy(deviceInput, hostInput, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input to device");

    dim3 blockSize(32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    prefixsum_kernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N);
    checkCudaError("Kernel launch failed");
    cudaDeviceSynchronize();

    cudaMemcpy(hostOutput, deviceOutput, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output to host");

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Print input and output arrays
    printf("\nInput Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", hostInput[i]);
    }

    printf("\n\nPrefix Sum Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", hostOutput[i]);
    }

    printf("\n");

    return 0;
}
