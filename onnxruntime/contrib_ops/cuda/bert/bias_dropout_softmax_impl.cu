// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_dropout_softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// duplicated softmax_impl.cu here to avoid hacking common case
// refer to additional comments in original source

// Note: The intended case for 'input_bias' is the input sequence mask for transformer models
// As an additive mask, it should be zero for preserved tokens and -infty for tokens to screen
// The mask will broadcast from [batch_size, 1, 1, seq_len] to input [batch_size, num_heads, seq_len, seq_len]
// Here element_count = seq_len and bias_broadcast_size_per_batch = num_heads * seq_len

// The softmax + additive mask fusion follows NVIDIA apex's additive_masked_softmax_warp_forward
// see https://github.com/NVIDIA/apex/blob/master/apex/contrib/csrc/multihead_attn/softmax.h

template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void bias_softmax_warp_forward(
  output_t* output, 
  const input_t* input, 
  const input_t* input_bias,
  int element_count,
  int batch_count, 
  int batch_stride,  
  int bias_broadcast_size_per_batch) {

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
  int local_idx = threadIdx.x;

  // push input, input_bias output pointers to batch we need to process
  input  += first_batch * batch_stride + local_idx;
  output += first_batch * batch_stride + local_idx;

  // load from global memory and apply bias (likely an additive mask)
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {

    // the bias has assumed shape [batch_size, element_count] 
    // .. and needs to broadcast to [batch_size, broadcast_size, element_count]
    int bias_offset = (first_batch + i)/bias_broadcast_size_per_batch * batch_stride + local_idx; 

    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = (input_t)src[i * element_count + it * WARP_SIZE] + (input_t)input_bias[mask_offset + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // find maximum value within batch for numerical stability
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  // normalization factor Z = Sum[ exp(element_i), for element_i in batch ]
  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// write back normalized value = exp(element_i)/Z to global memory
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        output[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_bias_softmax_forward(
  output_t* output, 
  const input_t* input, 
  const input_t* input_bias, 
  int element_count, 
  int batch_count, 
  int batch_stride, 
  int bias_broadcast_size_per_batch) {
  if (element_count == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(element_count);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        bias_softmax_warp_forward<input_t, output_t, acc_t, 0>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 1:  // 2
        bias_softmax_warp_forward<input_t, output_t, acc_t, 1>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 2:  // 4
        bias_softmax_warp_forward<input_t, output_t, acc_t, 2>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 3:  // 8
        bias_softmax_warp_forward<input_t, output_t, acc_t, 3>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 4:  // 16
        bias_softmax_warp_forward<input_t, output_t, acc_t, 4>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 5:  // 32
        bias_softmax_warp_forward<input_t, output_t, acc_t, 5>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 6:  // 64
        bias_softmax_warp_forward<input_t, output_t, acc_t, 6>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 7:  // 128
        bias_softmax_warp_forward<input_t, output_t, acc_t, 7>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 8:  // 256
        bias_softmax_warp_forward<input_t, output_t, acc_t, 8>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 9:  // 512
        bias_softmax_warp_forward<input_t, output_t, acc_t, 9>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      case 10:  // 1024
        bias_softmax_warp_forward<input_t, output_t, acc_t, 10>
            <<<blocks, threads, 0>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
        break;
      default:
        break;
    }
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
