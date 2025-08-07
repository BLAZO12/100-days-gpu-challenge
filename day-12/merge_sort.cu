#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__device__ void co_rank_debug(const int* input_array1, const int* input_array2, int k, const int size1, const int size2, int* i_out, int* j_out) {
    int low = max(0, k - size2);
    int high = min(k, size1);

    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;

        if (j < 0) {
            high = i - 1;
            continue;
        }
        if (j > size2) {
            low = i + 1;
            continue;
        }

        if (i > 0 && j < size2 && input_array1[i - 1] > input_array2[j]) {
            high = i - 1;
        } else if (j > 0 && i < size1 && input_array2[j - 1] > input_array1[i]) {
            low = i + 1;
        } else {
            *i_out = i;
            *j_out = j;

            // Debug print (limit for first few threads)
            if (k < 10) {
                printf("co_rank[k=%d] => i=%d, j=%d\n", k, i, j);
            }
            return;
        }
    }
}

__global__ void parallel_merge_debug(const int* input_array1, const int* input_array2, int* merged_result, const int size1, const int size2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size1 + size2) {
        int i, j;
        co_rank_debug(input_array1, input_array2, tid, size1, size2, &i, &j);

        if (j >= size2 || (i < size1 && input_array1[i] <= input_array2[j])) {
            merged_result[tid] = input_array1[i];
            if (tid < 10) printf("TID %d: picked %d from input_array1[%d]\n", tid, input_array1[i], i);
        } else {
            merged_result[tid] = input_array2[j];
            if (tid < 10) printf("TID %d: picked %d from input_array2[%d]\n", tid, input_array2[j], j);
        }
    }
}

int main() {
    const int size1 = 5;
    const int size2 = 5;
    int input_array1[size1], input_array2[size2], merged_result[size1 + size2];

    // Initialize sorted arrays
    for (int i = 0; i < size1; i++) {
        input_array1[i] = 2 * i; // 0, 2, 4, 6, 8
    }
    for (int i = 0; i < size2; i++) {
        input_array2[i] = 2 * i + 1; // 1, 3, 5, 7, 9
    }

    printf("Input Array 1: ");
    for (int i = 0; i < size1; i++) printf("%d ", input_array1[i]);
    printf("\n");

    printf("Input Array 2: ");
    for (int i = 0; i < size2; i++) printf("%d ", input_array2[i]);
    printf("\n");

    // Device memory
    int *d_input_array1, *d_input_array2, *d_merged_result;

    cudaMalloc(&d_input_array1, size1 * sizeof(int));
    cudaMalloc(&d_input_array2, size2 * sizeof(int));
    cudaMalloc(&d_merged_result, (size1 + size2) * sizeof(int));

    cudaMemcpy(d_input_array1, input_array1, size1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_array2, input_array2, size2 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(256);
    dim3 grid((size1 + size2 + block.x - 1) / block.x);

    printf("Launching kernel...\n");
    parallel_merge_debug<<<grid, block>>>(d_input_array1, d_input_array2, d_merged_result, size1, size2);
    cudaDeviceSynchronize();

    cudaMemcpy(merged_result, d_merged_result, (size1 + size2) * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input_array1);
    cudaFree(d_input_array2);
    cudaFree(d_merged_result);

    printf("Merged Result: ");
    for (int i = 0; i < size1 + size2; i++) {
        printf("%d ", merged_result[i]);
    }
    printf("\n");

    return 0;
}
