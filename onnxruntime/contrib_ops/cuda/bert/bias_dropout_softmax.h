// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename input_t, typename output_t, typename acc_t>
void dispatch_bias_softmax_forward(
  output_t* dst, 
  const input_t* src,
  const input_t* bias,
  int softmax_elements, 
  int batch_stride, 
  int batch_count,
  int bias_repeat_count);

template <typename T>
class BiasDropoutSoftmax final : public CudaKernel {
 public:
  BiasDropoutSoftmax(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
