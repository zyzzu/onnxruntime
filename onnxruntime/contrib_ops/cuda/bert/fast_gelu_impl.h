// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

#ifdef USE_CUDA_FP16
void launchGeluKernel(
    const void* input,         // input tensor
    void* output,              // output tensor
    const int element_count,   // number of elements in input tensor
    const size_t element_size  // element size of input tensor
);
#endif
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
