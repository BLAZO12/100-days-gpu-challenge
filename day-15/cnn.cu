#include <cuda_runtime.h>
#include <iostream>

#define KERNEL_SIZE 3
#define INPUT_CHANNELS 3
#define OUTPUT_CHANNELS 2
#define INPUT_WIDTH 5
#define INPUT_HEIGHT 5
#define OUTPUT_WIDTH (INPUT_WIDTH - KERNEL_SIZE + 1)
#define OUTPUT_HEIGHT (INPUT_HEIGHT - KERNEL_SIZE + 1)

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void conv_input_grad_kernel(float* d_input, const float* d_output, const float* weights,
                                       int in_channels, int out_channels,
                                       int input_width, int input_height,
                                       int output_width, int output_height,
                                       int kernel_size) {
    int ic = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input_width || y >= input_height || ic >= in_channels) return;

    float value = 0.0f;
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int out_x = x - i;
                int out_y = y - j;
                if (out_x >= 0 && out_x < output_width && out_y >= 0 && out_y < output_height) {
                    int out_idx = oc * output_width * output_height + out_y * output_width + out_x;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + i * kernel_size + j;
                    value += d_output[out_idx] * weights[weight_idx];
                }
            }
        }
    }
    int input_idx = ic * input_width * input_height + y * input_width + x;
    d_input[input_idx] = value;

    printf("input_grad[%d][%d][%d] = %f\n", ic, y, x, value);
}

__global__ void conv_weight_grad_kernel(float* d_weights, const float* d_output, const float* input,
                                        int in_channels, int out_channels,
                                        int input_width, int input_height,
                                        int output_width, int output_height,
                                        int kernel_size) {
    int oc = blockIdx.z;
    int ic = blockIdx.y;
    int kx = threadIdx.x;
    int ky = threadIdx.y;

    if (kx >= kernel_size || ky >= kernel_size || ic >= in_channels || oc >= out_channels) return;

    float value = 0.0f;
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            int out_idx = oc * output_width * output_height + y * output_width + x;
            int in_x = x + kx;
            int in_y = y + ky;
            int input_idx = ic * input_width * input_height + in_y * input_width + in_x;
            value += d_output[out_idx] * input[input_idx];
        }
    }
    int weight_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + ky * kernel_size + kx;
    d_weights[weight_idx] = value;

    printf("weight_grad[%d][%d][%d][%d] = %f\n", oc, ic, ky, kx, value);
}

void launch_convolution_backward_kernels(float* d_input, float* d_weights, const float* d_output, const float* input, const float* weights) {
    dim3 blockDim_input(16, 16);
    dim3 gridDim_input((INPUT_WIDTH + 15) / 16, (INPUT_HEIGHT + 15) / 16, INPUT_CHANNELS);
    conv_input_grad_kernel<<<gridDim_input, blockDim_input>>>(d_input, d_output, weights,
        INPUT_CHANNELS, OUTPUT_CHANNELS,
        INPUT_WIDTH, INPUT_HEIGHT,
        OUTPUT_WIDTH, OUTPUT_HEIGHT,
        KERNEL_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    dim3 blockDim_weight(KERNEL_SIZE, KERNEL_SIZE);
    dim3 gridDim_weight(1, INPUT_CHANNELS, OUTPUT_CHANNELS);
    conv_weight_grad_kernel<<<gridDim_weight, blockDim_weight>>>(d_weights, d_output, input,
        INPUT_CHANNELS, OUTPUT_CHANNELS,
        INPUT_WIDTH, INPUT_HEIGHT,
        OUTPUT_WIDTH, OUTPUT_HEIGHT,
        KERNEL_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

int main() {
    float *d_input, *d_weights, *d_output, *input, *weights;
    size_t input_size = INPUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float);
    size_t output_size = OUTPUT_CHANNELS * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(float);
    size_t weight_size = OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_weights, weight_size);

    cudaMalloc(&input, input_size);
    cudaMalloc(&weights, weight_size);

    // Fill d_output, input, and weights with dummy data if needed here

    launch_convolution_backward_kernels(d_input, d_weights, d_output, input, weights);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(input);
    cudaFree(weights);

    return 0;
}
