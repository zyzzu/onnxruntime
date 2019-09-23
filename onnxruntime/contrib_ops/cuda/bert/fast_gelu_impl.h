/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// NVIDIA/apex is licensed under the
// BSD 3 - Clause "New" or "Revised" License
//

/* Modifications Copyright (c) Microsoft. */

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
