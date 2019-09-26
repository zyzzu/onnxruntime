/*
 The implementation of this file is based on skipLayerNorm plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/
 
Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm.cuh"
#include "skip_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

/*
 It uses FP16 functions (like hrsqrt and __hadd2), which are only supported on arch >= 5.3
*/
#ifdef USE_CUDA_FP16

template <typename T, unsigned TPB>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const float* beta, const float* gamma, T* output) {
  const T reverse_ld = T(1) / T(ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pairSum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  if (threadIdx.x < ld) {
    val = input[idx] + skip[idx];
    const T rldval = reverse_ld * val;
    thread_data = pairSum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  layerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB>
__global__ void skipLayerNormKernel(
    const int ld, const T* input, const T* skip, const float* beta, const float* gamma, T* output) {
  const T reverse_ld = T(1) / T(ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pairSum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = input[idx] + skip[idx];
    const T rldval = reverse_ld * val;
    thread_data = pairSum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
    output[idx] = val;
  }

  layerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, output);
}

template <typename T>
void computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* skip,
                          const float* beta, const float* gamma, T* output) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int grid_size = n / ld;

  if (ld <= 32) {
    constexpr int block_size = 32;
    skipLayerNormKernelSmall<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, output);
  } else if (ld <= 128) {
    constexpr int block_size = 128;
    skipLayerNormKernelSmall<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, output);
  } else if (ld == 384) {
    constexpr int block_size = 384;
    skipLayerNormKernelSmall<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, output);
  } else {
    constexpr int block_size = 256;
    skipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, output);
  }
  CUDA_CALL(cudaPeekAtLastError());
}

void launchSkipLayerNormKernel(
    void* output,
    const void* input,
    const void* skip,
    const float* gamma,
    const float* beta,
    const int batch_size,
    const int hidden_size,
    const int element_count,
    const size_t element_size) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    computeSkipLayerNorm(stream, hidden_size, element_count,
                         reinterpret_cast<const half*>(input), reinterpret_cast<const half*>(skip),
                         beta, gamma, reinterpret_cast<half*>(output));
  } else {
    computeSkipLayerNorm(stream, hidden_size, element_count,
                         reinterpret_cast<const float*>(input), reinterpret_cast<const float*>(skip),
                         beta, gamma, reinterpret_cast<float*>(output));
  }
}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
