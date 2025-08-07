#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define FHD_THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

using namespace cv;
using namespace std;

__constant__ float kx_c[CHUNK_SIZE], ky_c[CHUNK_SIZE], kz_c[CHUNK_SIZE];

__global__ void cmpFHd(float* realPhi, float* imagPhi, float* phiMagnitude,
                       float* coordX, float* coordY, float* intensityZ,
                       float* realMu, float* imagMu, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    
    float xVal = coordX[n]; 
    float yVal = coordY[n]; 
    float zVal = intensityZ[n];

    float realOut = realPhi[n]; 
    float imagOut = imagPhi[n];

    for (int m = 0; m < M; m++) {
        float expVal = 2 * PI * (kx_c[m] * xVal + ky_c[m] * yVal + kz_c[m] * zVal);
        float cosPart = __cosf(expVal);
        float sinPart = __sinf(expVal);

        realOut += realMu[m] * cosPart - imagMu[m] * sinPart;
        imagOut += imagMu[m] * cosPart + realMu[m] * sinPart;
    }

    realPhi[n] = realOut;
    imagPhi[n] = imagOut;
}

int main() {
    Mat image = imread("lena_gray.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Could not open the image!" << endl;
        return -1;
    }

    cout << "Image loaded: " << image.cols << "x" << image.rows << endl;

    image.convertTo(image, CV_32F, 1.0 / 255);

    int N = image.rows * image.cols;
    int M = 256;

    float *coordX, *coordY, *intensityZ;
    float *realMu, *imagMu;
    float *realPhi, *imagPhi, *phiMagnitude;

    cudaMallocManaged(&coordX, N * sizeof(float));
    cudaMallocManaged(&coordY, N * sizeof(float));
    cudaMallocManaged(&intensityZ, N * sizeof(float));
    cudaMallocManaged(&realMu, M * sizeof(float));
    cudaMallocManaged(&imagMu, M * sizeof(float));
    cudaMallocManaged(&realPhi, N * sizeof(float));
    cudaMallocManaged(&imagPhi, N * sizeof(float));
    cudaMallocManaged(&phiMagnitude, N * sizeof(float));

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            coordX[idx] = (float)j / image.cols;
            coordY[idx] = (float)i / image.rows;
            intensityZ[idx] = image.at<float>(i, j);
            realPhi[idx] = intensityZ[idx];
            imagPhi[idx] = 0.0f;
        }
    }

    for (int i = 0; i < M; i++) {
        realMu[i] = static_cast<float>(rand()) / RAND_MAX;
        imagMu[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Print a few samples
    cout << "Sample input coordinates and intensities:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << "coordX[" << i << "]=" << coordX[i]
             << ", coordY[" << i << "]=" << coordY[i]
             << ", intensityZ[" << i << "]=" << intensityZ[i] << endl;
    }

    cout << "Sample realMu and imagMu:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << "realMu[" << i << "]=" << realMu[i]
             << ", imagMu[" << i << "]=" << imagMu[i] << endl;
    }

    for (int i = 0; i < M / CHUNK_SIZE; i++) {
        cudaMemcpyToSymbol(kx_c, &coordX[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(ky_c, &coordY[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(kz_c, &intensityZ[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        cmpFHd<<<N / FHD_THREADS_PER_BLOCK, FHD_THREADS_PER_BLOCK>>>(
            realPhi, imagPhi, phiMagnitude,
            coordX, coordY, intensityZ,
            realMu, imagMu, CHUNK_SIZE);
        
        cudaDeviceSynchronize();
        cout << "Kernel chunk " << i + 1 << " executed." << endl;
    }

    Mat outputImage(image.rows, image.cols, CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            outputImage.at<float>(i, j) = sqrt(realPhi[idx] * realPhi[idx] + imagPhi[idx] * imagPhi[idx]);
        }
    }

    // Sample output
    cout << "Sample output magnitude:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << "phiMagnitude[" << i << "] = " << sqrt(realPhi[i] * realPhi[i] + imagPhi[i] * imagPhi[i]) << endl;
    }

    normalize(outputImage, outputImage, 0, 255, NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8U);
    imwrite("output.jpg", outputImage);

    cudaFree(coordX);
    cudaFree(coordY);
    cudaFree(intensityZ);
    cudaFree(realMu);
    cudaFree(imagMu);
    cudaFree(realPhi);
    cudaFree(imagPhi);
    cudaFree(phiMagnitude);

    cout << "Processed image saved as output.jpg" << endl;
    return 0;
}
