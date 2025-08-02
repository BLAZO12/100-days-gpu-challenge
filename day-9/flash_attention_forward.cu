
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define sequence_length 8
#define embed_dimension 4
#define BLOCK_SIZE 4
#define attention_scale (1.0f / sqrtf((float)embed_dimension))

__global__ void flashAttentionForward(
    const float *q_matrix,
    const float *k_matrix,
    const float *v_matrix,
    float *out_matrix,
    float *max_per_row,
    float *sum_per_row,
    const float scale
) {
    __shared__ float Query_block[BLOCK_SIZE * embed_dimension];
    __shared__ float Key_block[BLOCK_SIZE * embed_dimension];
    __shared__ float Value_block[BLOCK_SIZE * embed_dimension];
    __shared__ float attention_scores[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float attention_weights[BLOCK_SIZE * BLOCK_SIZE];

    int thread_idx = threadIdx.x;
    int row_block = blockIdx.x;

    if (thread_idx < BLOCK_SIZE) {
        for (int d = 0; d < embed_dimension; ++d) {
            Query_block[thread_idx * embed_dimension + d] =
                q_matrix[(row_block * BLOCK_SIZE + thread_idx) * embed_dimension + d];
            Key_block[thread_idx * embed_dimension + d] =
                k_matrix[(thread_idx) * embed_dimension + d];
            Value_block[thread_idx * embed_dimension + d] =
                v_matrix[(thread_idx) * embed_dimension + d];
        }
    }
    __syncthreads();

    if (thread_idx < BLOCK_SIZE) {
        float row_max = -1e20f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float score = 0.0f;
            for (int d = 0; d < embed_dimension; ++d) {
                score += Query_block[thread_idx * embed_dimension + d] *
                         Key_block[k * embed_dimension + d];
            }
            score *= scale;
            attention_scores[thread_idx * BLOCK_SIZE + k] = score;
            row_max = fmaxf(row_max, score);
            printf("[Block %d Thread %d] score[%d,%d] = %.5f\n", blockIdx.x, thread_idx, row_block, k, score);
        }

        float row_sum = 0.0f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float weight = expf(attention_scores[thread_idx * BLOCK_SIZE + k] - row_max);
            attention_weights[thread_idx * BLOCK_SIZE + k] = weight;
            row_sum += weight;
        }

        max_per_row[row_block * BLOCK_SIZE + thread_idx] = row_max;
        sum_per_row[row_block * BLOCK_SIZE + thread_idx] = row_sum;
        printf("[Block %d Thread %d] row_max = %.5f, row_sum = %.5f\n", blockIdx.x, thread_idx, row_max, row_sum);

        for (int d = 0; d < embed_dimension; ++d) {
            float weighted_sum = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                weighted_sum += attention_weights[thread_idx * BLOCK_SIZE + k] *
                                Value_block[k * embed_dimension + d];
            }
            float result = (row_sum > 0) ? (weighted_sum / row_sum) : 0.0f;
            out_matrix[(row_block * BLOCK_SIZE + thread_idx) * embed_dimension + d] = result;
            printf("[Block %d Thread %d] out[%d][%d] = %.5f\n", blockIdx.x, thread_idx,
                   row_block * BLOCK_SIZE + thread_idx, d, result);
        }
    }
}

int main() {
    float *q_matrix = new float[sequence_length * embed_dimension];
    float *k_matrix = new float[sequence_length * embed_dimension];
    float *v_matrix = new float[sequence_length * embed_dimension];
    float *out_matrix = new float[sequence_length * embed_dimension];
    float *max_per_row = new float[sequence_length];
    float *sum_per_row = new float[sequence_length];

    for (int i = 0; i < sequence_length * embed_dimension; ++i) {
        q_matrix[i] = (float)(i % 5 + 1);
        k_matrix[i] = (float)(i % 3 + 1);
        v_matrix[i] = (float)(i % 4 + 2);
    }

    float *d_q, *d_k, *d_v, *d_out, *d_max, *d_sum;
    cudaMalloc(&d_q, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&d_k, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&d_v, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&d_out, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&d_max, sequence_length * sizeof(float));
    cudaMalloc(&d_sum, sequence_length * sizeof(float));

    cudaMemcpy(d_q, q_matrix, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_matrix, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_matrix, sequence_length * embed_dimension * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(sequence_length / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    flashAttentionForward<<<grid, block>>>(d_q, d_k, d_v, d_out, d_max, d_sum, attention_scale);
    cudaDeviceSynchronize();

    cudaMemcpy(out_matrix, d_out, sequence_length * embed_dimension * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nFinal Output Matrix:\n");
    for (int i = 0; i < sequence_length; ++i) {
        printf("Position %d: ", i);
        for (int j = 0; j < embed_dimension; ++j) {
            printf("%.4f ", out_matrix[i * embed_dimension + j]);
        }
        printf("\n");
    }

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_max); cudaFree(d_sum);
    delete[] q_matrix; delete[] k_matrix; delete[] v_matrix;
    delete[] out_matrix; delete[] max_per_row; delete[] sum_per_row;

    return 0;
}
