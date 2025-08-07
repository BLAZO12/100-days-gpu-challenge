// NaiveBayes.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NaiveBayesKernel.cuh"
#include "NaiveBayesTrain.cuh"

#define SHARED_PRIORS 20
#define SHARED_LIKELIHOODS 2000  // Adjust size based on numClasses * numFeatures * numFeatureValues

// CUDA Kernel to compute priors (P(Y = c)) and likelihoods (P(X | Y = c)).
__global__ void computePriorsAndLikelihood(
    int* d_dataset, int* d_priors_output, int* d_likelihoods_output,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_priors[SHARED_PRIORS];
    __shared__ int shared_likelihoods[SHARED_LIKELIHOODS];

    // Initialize shared memory
    if (threadIdx.x < SHARED_PRIORS) {
        shared_priors[threadIdx.x] = 0;
    }
    if (threadIdx.x < SHARED_LIKELIHOODS) {
        shared_likelihoods[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadId < numSamples) {
        // Each thread processes one data sample
        int sampleStart = threadId * (numFeatures + 1);
        int classLabel = d_dataset[sampleStart + numFeatures]; // Class label is in the last column

        // Print sample info
        printf("Thread %d processing sample %d, class = %d\n", threadId, threadId, classLabel);

        // Atomic update to calculate prior
        atomicAdd(&shared_priors[classLabel], 1);

        // Compute likelihood for each feature
        for (int fIdx = 0; fIdx < numFeatures; ++fIdx) {
            int featureValue = d_dataset[sampleStart + fIdx];
            int likelihoodIndex = classLabel * numFeatures * numFeatureValues + (fIdx * numFeatureValues) + featureValue;

            printf("Thread %d: feature %d = %d, likelihoodIdx = %d\n", threadId, fIdx, featureValue, likelihoodIndex);

            // Atomic update to the likelihood matrix
            atomicAdd(&shared_likelihoods[likelihoodIndex], 1);
        }
    }

    __syncthreads();

    // Write shared results to global memory
    if (threadIdx.x == 0) {
        for (int c = 0; c < numClasses; ++c) {
            atomicAdd(&d_priors_output[c], shared_priors[c]);
        }

        for (int l = 0; l < numClasses * numFeatures * numFeatureValues; ++l) {
            atomicAdd(&d_likelihoods_output[l], shared_likelihoods[l]);
        }
    }
}
#include <iostream>
#include "NaiveBayesKernel.cuh"

int main() {
    std::cout << "Naive Bayes CUDA Program Started." << std::endl;

    // Define dummy variables to pass to kernel (allocate real ones later)
    int* d_dataset = nullptr;
    int* d_classCounts = nullptr;
    int* d_featureLikelihoods = nullptr;

    int numSamples = 100;
    int numFeatures = 10;
    int numClasses = 2;
    int numFeatureValues = 2;

    // Call the kernel with <<<1, 1>>> just for compilation (not real use)
    computePriorsAndLikelihood<<<1, 1>>>(
        d_dataset, d_classCounts, d_featureLikelihoods,
        numSamples, numFeatures, numClasses, numFeatureValues
    );

    cudaDeviceSynchronize();  // Ensure kernel completes

    std::cout << "Kernel launched successfully." << std::endl;
    return 0;
}

