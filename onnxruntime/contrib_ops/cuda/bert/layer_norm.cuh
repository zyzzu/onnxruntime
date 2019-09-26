/*
 The implementation of this file is based on bert plugins in TensorRT demo:
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
#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

/*
 It uses FP16 functions (like hrsqrt and __hadd2), which are only supported on arch >= 5.3
*/
#ifdef USE_CUDA_FP16

template <typename T>
__device__ inline T rsqrt(const T& x);

template <>
__device__ inline float rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline half rsqrt(const half& x) {
  return hrsqrt(x);
}

struct KeyValuePairSum {
  __device__ inline cub::KeyValuePair<float, float> operator()(const cub::KeyValuePair<float, float>& a, const cub::KeyValuePair<float, float>& b) {
    return cub::KeyValuePair<float, float>(a.key + b.key, a.value + b.value);
  }

  __device__ inline cub::KeyValuePair<half, half> operator()(const cub::KeyValuePair<half, half>& a, const cub::KeyValuePair<half, half>& b) {
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = __hadd2(a2, b2);
    return cub::KeyValuePair<half, half>(res.x, res.y);
  }

  __device__ inline cub::KeyValuePair<half2, half2> operator()(const cub::KeyValuePair<half2, half2>& a, const cub::KeyValuePair<half2, half2>& b) {
    return cub::KeyValuePair<half2, half2>(__hadd2(a.key, b.key), __hadd2(a.value, b.value));
  }
};

template <typename T, int TPB>
__device__ inline void layerNorm(
    const cub::KeyValuePair<T, T>& threadData, const int ld, const int offset, const float* beta, const float* gamma, T* output) {
  // Assuming threadData is already divided by ld

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  KeyValuePairSum pairSum;
  const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, pairSum);

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(gamma[i]);
    const T b(beta[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, int TPB>
__device__ inline void layerNormSmall(const T val, const cub::KeyValuePair<T, T>& threadData, const int ld, const int idx,
                                      const float* beta, const float* gamma, T* output) {
  // Assuming threadData is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  KeyValuePairSum pairSum;
  const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, pairSum);

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    const T g(gamma[threadIdx.x]);
    const T b(beta[threadIdx.x]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
