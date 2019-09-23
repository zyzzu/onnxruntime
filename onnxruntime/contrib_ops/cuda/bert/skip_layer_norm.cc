// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/framework/tensorprotoutils.h"
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

static void FetchDataFromTensor(TensorProto& t_proto, std::vector<float>& value) {
  ORT_ENFORCE(t_proto.has_data_type());
  ORT_ENFORCE(TensorProto::DataType_IsValid(t_proto.data_type()));
  const auto tensor_type = static_cast<TensorProto_DataType>(t_proto.data_type());
  const void* const raw_data = t_proto.has_raw_data() ? t_proto.raw_data().data() : nullptr;
  const size_t raw_data_len = t_proto.has_raw_data() ? t_proto.raw_data().size() : 0;

  int64_t expected_size = 1;
  for (int d = 0; d < t_proto.dims_size(); d++) expected_size *= t_proto.dims()[d];
  value.resize(expected_size);
  auto unpack_status = utils::UnpackTensor(t_proto, raw_data, raw_data_len, value.data(), expected_size);
  ORT_ENFORCE(unpack_status.IsOK(), "Value attribute unpacking failed:", unpack_status.ErrorMessage());
}

template <typename T>
SkipLayerNorm<T>::SkipLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  TensorProto t_proto;
  op_kernel_info.GetAttr<TensorProto>("gamma", &t_proto);
  FetchDataFromTensor(t_proto, gamma_);

  op_kernel_info.GetAttr<TensorProto>("beta", &t_proto);
  FetchDataFromTensor(t_proto, beta_);

  gamma_data_ = GetScratchBuffer<float>(gamma_.size());
  CUDA_CALL_THROW(cudaMemcpy(gamma_data_.get(), gamma_.data(), sizeof(float) * gamma_.size(), cudaMemcpyHostToDevice));

  beta_data_ = GetScratchBuffer<float>(beta_.size());
  CUDA_CALL_THROW(cudaMemcpy(beta_data_.get(), beta_.data(), sizeof(float) * beta_.size(), cudaMemcpyHostToDevice));
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

  if (gamma_.size() != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma shape does not match with input");
  }
  if (beta_.size() != input_dims[2]) {
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
