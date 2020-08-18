// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_dropout_softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      BiasDropoutSoftmax,                                                       \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BiasDropoutSoftmax<T>);                                                              

template <typename T>
Status BiasDropoutSoftmax<T>::ComputeInternal(OpKernelContext* ctx) const {
  std::string s = ctx->GetOpDomain();
  
  if (element_count <= 1024) {
    // presumably thread block still fits within SM registers at high occupancy
    // call dispatch_bias_softmax_forward()
  }
  else {
    // fallback to original routine - will dispatch to cuDNN
    // call softmax(...)
  }

  // call dropout_kernel(...)

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status BiasDropoutSoftmax<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
