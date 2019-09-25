// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

#ifdef USE_CUDA_FP16
void launchSkipLayerNormKernel(
    void* output,              // output tensor
    const void* input,         // input tensor
    const void* skip,          // skip tensor
    const float* gamma,        // weight tensor
    const float* beta,         // bias tensor
    const int batch_size,      // batch size (B)
    const int hidden_size,     // hidden size, it is the leading dimension (ld)
    const int element_count,   // number of elements in input tensor
    const size_t element_size  // element size of input tensor
);
#endif
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
