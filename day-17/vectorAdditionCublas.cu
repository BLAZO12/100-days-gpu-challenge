// nvcc vec_cublas.cu -o vec_cublas -lstdc++ -lcublas

#include <iostream>
#include <cublas_v2.h>

int main() {
    const int length = 10;
    float host_vectorA[length], host_vectorB[length], host_result[length];

    // Initialize input vectors
    std::cout << "Initializing host vectors...\n";
    for (int i = 0; i < length; i++) {
        host_vectorA[i] = static_cast<float>(i);
        host_vectorB[i] = static_cast<float>(i);
    }

    std::cout << "Vector A: ";
    for (int i = 0; i < length; i++) std::cout << host_vectorA[i] << " ";
    std::cout << "\n";

    std::cout << "Vector B: ";
    for (int i = 0; i < length; i++) std::cout << host_vectorB[i] << " ";
    std::cout << "\n";

    // Create cuBLAS handle
    std::cout << "Creating cuBLAS handle...\n";
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Allocate memory on device
    std::cout << "Allocating device memory...\n";
    float *device_vectorA, *device_vectorB;
    cudaMalloc(&device_vectorA, length * sizeof(float));
    cudaMalloc(&device_vectorB, length * sizeof(float));

    // Copy host vectors to device
    std::cout << "Copying vectors to device...\n";
    cudaMemcpy(device_vectorA, host_vectorA, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vectorB, host_vectorB, length * sizeof(float), cudaMemcpyHostToDevice);

    // Set scaling factor
    const float alpha = 1.0f;

    // Perform vector addition: device_vectorB = alpha * device_vectorA + device_vectorB
    std::cout << "Performing vector addition using cuBLAS saxpy...\n";
    cublasSaxpy(cublas_handle, length, &alpha, device_vectorA, 1, device_vectorB, 1);

    // Copy result back to host
    std::cout << "Copying result back to host...\n";
    cudaMemcpy(host_result, device_vectorB, length * sizeof(float), cudaMemcpyDeviceToHost);

    // Display result
    std::cout << "Result (A + B): ";
    for (int i = 0; i < length; i++) {
        std::cout << host_result[i] << " ";
    }
    std::cout << "\n";

    // Cleanup
    std::cout << "Freeing memory and destroying handle...\n";
    cudaFree(device_vectorA);
    cudaFree(device_vectorB);
    cublasDestroy(cublas_handle);

    std::cout << "Done.\n";
    return 0;
}
