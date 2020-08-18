// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_dropout_input_grad.h"

namespace onnxruntime {
namespace cuda {

// we duplicate code from softmax_grad_impl.cu to avoid hacking common path
// compute dX := d(loss)/dX for Z = dropout(softmax(X))
template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void dropout_softmax_warp_backward(
  output_t* input_grad, 
  const input_t* dropout_grad, 
  const input_t* softmax_output, 
  const bool *dropout_mask, 
  acc_t dropout_scaling, 
  int element_count,
  int batch_count, 
  int batch_stride) {

  // "WARP" refers to cooperative threads and might not equal 32 threads of GPU warp
  // thread block is (WARP_SIZE, 128/WARP_SIZE)
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  // each "WARP" (<=32) processes WARP_BATCH(one of {1,2}) batches
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // last warp may have fewer batches
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // thread will process elements (local_index + n * warp_size) within batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // push pointers to batch we need to process
  int thread_offset = first_batch * batch_stride + local_idx;
  dropout_grad += thread_offset;
  softmax_output += thread_offset;
  input_grad += thread_offset;

  // load data from global memory and apply dropout mask and scaling
  acc_t softmax_grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t softmax_output_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t softmax_grad_output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        softmax_grad_reg[i][it] = (input_t)( 
          (acc_t)dropout_mask[i*element_count+it*WARP_SIZE] *
          (acc_t)dropout_grad[i*element_count+it*WARP_SIZE] *
          (acc_t)dropout_scaling );

        softmax_output_reg[i][it] = output[i * element_count + it * WARP_SIZE];
        softmax_grad_output_reg[i][it] = grad_reg[i][it] * output_reg[i][it];
      } else {
        softmax_grad_reg[i][it] = acc_t(0);
        softmax_output_reg[i][it] = acc_t(0);
        softmax_grad_output_reg[i][it] = acc_t(0);
      }
    }
  }

  // normalization factor Z = Sum[ exp(element_i), for element_i in batch ]
  acc_t sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = softmax_grad_output_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += softmax_grad_output_reg[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// write back input gradient to global memory
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        input_grad[i * element_count + it * WARP_SIZE] = (softmax_grad_reg[i][it] - sum[i]) * softmax_output_reg[i][it];
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_dropout_softmax_warp_backward(
  output_t* input_grad, 
  const input_t* dropout_grad, 
  const input_t* softmax_output, 
  const bool *dropout_mask, 
  acc_t dropout_ratio, 
  int element_count,
  int batch_count, 
  int batch_stride) {
  if (element_count == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    acc_t dropout_scaling = (acc_t)(1.0)/(acc_t(1.0) - dropout_ratio);

    switch (log2_elements) {
      case 0:  // 1
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 1:  // 2
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 2:  // 4
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 3:  // 8
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 4:  // 16
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 5:  // 32
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 6:  // 64
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 7:  // 128
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 8:  // 256
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 9:  // 512
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      case 10:  // 1024
      dropout_softmax_warp_backward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0>>>(input_grad, dropout_grad, dropout_scaling, element_count, batch_count, batch_stride);
        break;
      default:
        break;
    }
  }
}

} // namespace cuda
} // namespace onnxruntime