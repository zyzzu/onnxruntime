// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "skip_layer_norm.h"
#include "skip_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      SkipLayerNormalization,                                    \
      kOnnxDomain,                                               \
      1,                                                         \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),\
      SkipLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;


#define COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(attribute)                                                 \
  TensorProto attribute##_proto;                                                              \
  op_kernel_info.GetAttr<TensorProto>(#attribute, &attribute##_proto);                        \
  ORT_ENFORCE(attribute##_proto.data_type() == TensorProto::FLOAT);                           \
  std::vector<float> attribute##_data = ONNX_NAMESPACE::ParseData<float>(&attribute##_proto); \
  attribute##_size_ = attribute##_data.size();                                                \
  attribute##_data_ = GetScratchBuffer<float>(attribute##_size_);                             \
  CUDA_CALL_THROW(cudaMemcpy(attribute##_data_.get(), attribute##_data.data(), sizeof(float) * attribute##_size_, cudaMemcpyHostToDevice))

template <typename T>
SkipLayerNorm<T>::SkipLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(gamma);
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(beta);
}

template <typename T>
Status SkipLayerNorm<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* skip = ctx->Input<Tensor>(1);
  Tensor* output = ctx->Output(0, input->Shape());

  const auto input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 dimensions, got ", input_dims.size());
  }

  if (input->Shape() != skip->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "skip is expected to have same shape as input");
  }

  if (gamma_size_ != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma shape does not match with input");
  }
  if (beta_size_ != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta shape does not match with input");
  }

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  int hidden_size = static_cast<int>(input_dims[2]);
  int element_count = batch_size * sequence_length * hidden_size;
  size_t element_size = sizeof(T);

  #ifdef USE_CUDA_FP16
  launchSkipLayerNormKernel(
    output->template MutableData<T>(),
    input->template Data<T>(),
    skip->template Data<T>(),
    gamma_data_.get(),
    beta_data_.get(),
    batch_size,
    hidden_size,
    element_count,
    element_size);
#endif

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
