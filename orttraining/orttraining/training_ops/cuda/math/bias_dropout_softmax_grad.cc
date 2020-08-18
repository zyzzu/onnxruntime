// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_dropout_softmax_grad.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      BiasDropoutSoftmaxGrad_dX,                                                \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BiasDropoutSoftmaxGrad_dX<T>);

Status BiasDropoutSoftmaxGrad_dX<T>::ComputeInternal(OpKernelContext* ctx) const {
  std::string s = ctx->GetOpDomain();

  if (element_count <= 1024) {
    // presumably thread block fits within SM registers at high occupancy
    // call dispatch_dropout_softmax_backward(...)
  }
  else {
    // fallback to original routines - which dispatch to cuDNN
    // call dropout_grad(...)
    // call dispatch_softmax_backward(...)
  }

  return Status::OK();
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status BiasDropoutSoftmaxGrad_dX<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)

} // namespace cuda
} // namespace onnxruntime