#include <stdio.h>

__global__ void partialSumKernel(int *input, int *output, int numElements) {
    extern __shared__ int sharedMemory[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + tid;

    if (index + blockDim.x < numElements) {
        // Load two elements at a time (coalesced)
        sharedMemory[tid] = input[index] + input[index + blockDim.x];
    } else if (index < numElements) {
        sharedMemory[tid] = input[index];  // only one element left
    } else {
        sharedMemory[tid] = 0;  // out of bounds
    }

    __syncthreads();

    // Inclusive scan (prefix sum)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = sharedMemory[tid - stride];
        }
        __syncthreads();
        sharedMemory[tid] += temp;
        __syncthreads();
    }

    if (index < numElements) {
        output[index] = sharedMemory[tid];
    }
}

int main() {
    const int numElements = 16;
    const int blockSize = 8;
    const int gridSize = numElements / blockSize;

    int hostInput[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int hostOutput[numElements];

    int *devInput, *devOutput;
    size_t byteSize = numElements * sizeof(int);

    printf("Allocating device memory...\n");
    cudaMalloc(&devInput, byteSize);
    cudaMalloc(&devOutput, byteSize);

    printf("Copying input array to device...\n");
    cudaMemcpy(devInput, hostInput, byteSize, cudaMemcpyHostToDevice);

    printf("Launching kernel with gridSize=%d, blockSize=%d...\n", gridSize, blockSize);
    partialSumKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(devInput, devOutput, numElements);
    cudaDeviceSynchronize();

    printf("Copying output array back to host...\n");
    cudaMemcpy(hostOutput, devOutput, byteSize, cudaMemcpyDeviceToHost);

    printf("\nInput Array:\n");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", hostInput[i]);
    }

    printf("\nOutput Array (Prefix Sum Approximation):\n");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", hostOutput[i]);
    }
    printf("\n");

    printf("Freeing device memory...\n");
    cudaFree(devInput);
    cudaFree(devOutput);

    printf("Done.\n");
    return 0;
}
