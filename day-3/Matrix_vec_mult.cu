#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorMatrixMult(const float* matrix, const float* vector, float* result, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size) {
        float sum = 0.0f;
        for (int col = 0; col < size; col++) {
            sum += matrix[row * size + col] * vector[col];
        }
        result[row] = sum;
    }
}

int main() {
    const int size = 10;
    float *hostMatrix, *hostVector, *hostResult;

    std::cout << "Allocating host memory...\n";
    hostMatrix = (float *)malloc(size * size * sizeof(float));
    hostVector = (float *)malloc(size * sizeof(float));
    hostResult = (float *)malloc(size * sizeof(float));

    std::cout << "Initializing host data...\n";
    for (int i = 0; i < size; i++) {
        hostVector[i] = 2.0f;
        hostResult[i] = 0.0f;
        for (int j = 0; j < size; j++) {
            hostMatrix[i * size + j] = 1.0f;
        }
    }

    float *devMatrix, *devVector, *devResult;

    std::cout << "Allocating device memory...\n";
    cudaMalloc(&devMatrix, size * size * sizeof(float));
    cudaMalloc(&devVector, size * sizeof(float));
    cudaMalloc(&devResult, size * sizeof(float));

    std::cout << "Copying data to device...\n";
    cudaMemcpy(devMatrix, hostMatrix, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devVector, hostVector, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    std::cout << "Launching kernel...\n";
    vectorMatrixMult<<<gridSize, blockSize>>>(devMatrix, devVector, devResult, size);
    cudaDeviceSynchronize();

    std::cout << "Copying result back to host...\n";
    cudaMemcpy(hostResult, devResult, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nMatrix A:\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f ", hostMatrix[i * size + j]);
        }
        printf("\n");
    }

    std::cout << "\nVector B:\n";
    for (int i = 0; i < size; i++) {
        printf("%.2f ", hostVector[i]);
    }
    printf("\n");

    std::cout << "\nResult Vector C (Matrix x Vector):\n";
    for (int i = 0; i < size; i++) {
        printf("%.2f ", hostResult[i]);
    }
    printf("\n");

    std::cout << "Freeing device and host memory...\n";
    cudaFree(devMatrix);
    cudaFree(devVector);
    cudaFree(devResult);
    free(hostMatrix);
    free(hostVector);
    free(hostResult);

    std::cout << "Done.\n";
    return 0;
}
