// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename input_t, typename output_t, typename acc_t>
void dispatch_dropout_softmax_backward(
  output_t* softmax_input_grad, 
  const input_t* dropout_output_grad,
  const input_t* softmax_output,
  const bool* dropout_mask,
  acc_t dropout_ratio,
  int softmax_elements, 
  int batch_stride, 
  int batch_count);

// derivative of bias (if required) is handled by gradient builder
class BiasDropoutSoftmaxGrad_dX final : public CudaKernel {
 public:
  BiasDropoutSoftmaxGrad_dX(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("softmax_axis", &softmax_axis_, static_cast<int64_t>(1));
    info.GetAttrOrDefault("broadcast_axis", &softmax_axis_, static_cast<int64_t>(0));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t softmax_axis_;
  int64_t broadcast_axis_;
};

} // namespace cuda
} // namespace onnxruntime