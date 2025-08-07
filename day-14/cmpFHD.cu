#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <random>
#include <iostream>

#define THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

__constant__ float dev_kx[CHUNK_SIZE], dev_ky[CHUNK_SIZE], dev_kz[CHUNK_SIZE];

__global__ void computeField(
    float* realPhi, float* imagPhi, float* magnitudePhi,
    float* coordX, float* coordY, float* coordZ,
    float* realMu, float* imagMu, int chunkLength
) {
    int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    float xVal = coordX[idx];
    float yVal = coordY[idx];
    float zVal = coordZ[idx];

    float realSum = realPhi[idx];
    float imagSum = imagPhi[idx];

    for (int j = 0; j < chunkLength; j++) {
        float arg = 2 * PI * (dev_kx[j] * xVal + dev_ky[j] * yVal + dev_kz[j] * zVal);
        float cosArg = __cosf(arg);
        float sinArg = __sinf(arg);

        realSum += realMu[j] * cosArg - imagMu[j] * sinArg;
        imagSum += imagMu[j] * cosArg + realMu[j] * sinArg;
    }

    realPhi[idx] = realSum;
    imagPhi[idx] = imagSum;
    magnitudePhi[idx] = sqrtf(realSum * realSum + imagSum * imagSum);
}

int main() {
    const int numPoints = 1024;
    const int numSources = 1024;

    std::cout << "🔧 Initializing CUDA simulation...\n";

    float *hostX, *hostY, *hostZ;
    float *realMu, *imagMu;
    float *realPhi, *imagPhi, *magnitudePhi;

    cudaMallocManaged(&hostX, numPoints * sizeof(float));
    cudaMallocManaged(&hostY, numPoints * sizeof(float));
    cudaMallocManaged(&hostZ, numPoints * sizeof(float));
    cudaMallocManaged(&realMu, numSources * sizeof(float));
    cudaMallocManaged(&imagMu, numSources * sizeof(float));
    cudaMallocManaged(&realPhi, numPoints * sizeof(float));
    cudaMallocManaged(&imagPhi, numPoints * sizeof(float));
    cudaMallocManaged(&magnitudePhi, numPoints * sizeof(float));

    std::cout << "✅ Memory allocated successfully.\n";

    // Initialize random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < numPoints; i++) {
        hostX[i] = dist(gen);
        hostY[i] = dist(gen);
        hostZ[i] = dist(gen);
        realPhi[i] = 0.0f;
        imagPhi[i] = 0.0f;
        magnitudePhi[i] = 0.0f;
    }

    for (int i = 0; i < numSources; i++) {
        realMu[i] = dist(gen);
        imagMu[i] = dist(gen);
    }

    std::cout << "📊 Data initialized.\n";
    std::cout << "🔍 Sample positions and sources:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "  Point " << i << ": (x=" << hostX[i] << ", y=" << hostY[i] << ", z=" << hostZ[i] << ")\n";
    }

    int totalChunks = numSources / CHUNK_SIZE;

    for (int chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        std::cout << "\n🚀 Launching chunk " << chunkIdx + 1 << "/" << totalChunks << "\n";

        cudaMemcpyToSymbol(dev_kx, &hostX[chunkIdx * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(dev_ky, &hostY[chunkIdx * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(dev_kz, &hostZ[chunkIdx * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        computeField<<<numPoints / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            realPhi, imagPhi, magnitudePhi,
            hostX, hostY, hostZ,
            realMu, imagMu,
            CHUNK_SIZE
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "❌ Kernel launch failed: " << cudaGetErrorString(err) << "\n";
            return 1;
        }

        cudaDeviceSynchronize();
    }

    std::cout << "\n✅ Computation finished. First 5 results:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  realPhi[" << i << "] = " << realPhi[i]
                  << ", imagPhi[" << i << "] = " << imagPhi[i]
                  << ", |Phi| = " << magnitudePhi[i] << "\n";
    }

    // Cleanup
    cudaFree(hostX); cudaFree(hostY); cudaFree(hostZ);
    cudaFree(realMu); cudaFree(imagMu);
    cudaFree(realPhi); cudaFree(imagPhi); cudaFree(magnitudePhi);

    std::cout << "\n🎉 Program completed successfully.\n";
    return 0;
}
