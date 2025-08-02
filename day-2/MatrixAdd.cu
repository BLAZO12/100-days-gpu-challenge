#include <iostream>
#include <cuda_runtime.h>

__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int matrixSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row >= matrixSize) || (col >= matrixSize)) return;

    C[row * matrixSize + col] = A[row * matrixSize + col] + B[row * matrixSize + col];
}

int main() {
    const int matrixSize = 10;
    float *hostA, *hostB, *hostC;

    std::cout << "Allocating host memory...\n";
    hostA = (float *)malloc(matrixSize * matrixSize * sizeof(float));
    hostB = (float *)malloc(matrixSize * matrixSize * sizeof(float));
    hostC = (float *)malloc(matrixSize * matrixSize * sizeof(float));

    std::cout << "Initializing host matrices...\n";
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            hostA[i * matrixSize + j] = 1.0f;
            hostB[i * matrixSize + j] = 2.0f;
            hostC[i * matrixSize + j] = 0.0f;
        }
    }

    float *devA, *devB, *devC;

    std::cout << "Allocating device memory...\n";
    cudaMalloc((void**)&devA, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&devB, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&devC, matrixSize * matrixSize * sizeof(float));

    std::cout << "Copying data to device...\n";
    cudaMemcpy(devA, hostA, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 16);
    dim3 gridDim(ceil(matrixSize / 32.0f), ceil(matrixSize / 16.0f));

    std::cout << "Launching kernel...\n";
    MatrixAdd_B<<<gridDim, blockDim>>>(devA, devB, devC, matrixSize);
    cudaDeviceSynchronize();

    std::cout << "Copying result back to host...\n";
    cudaMemcpy(hostC, devC, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nResult Matrix C:\n";
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%.2f ", hostC[i * matrixSize + j]);
        }
        printf("\n");
    }

    std::cout << "\nMatrix A:\n";
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%.2f ", hostA[i * matrixSize + j]);
        }
        printf("\n");
    }

    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%.2f ", hostB[i * matrixSize + j]);
        }
        printf("\n");
    }

    std::cout << "Freeing memory...\n";
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(hostA);
    free(hostB);
    free(hostC);

    std::cout << "Done.\n";
    return 0;
}
