// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "embed_layer_norm.h"
#include "embed_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormalization,                                    \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNorm<T>);

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
EmbedLayerNorm<T>::EmbedLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  TensorProto t_proto;
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(gamma);
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(beta);
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(word_embedding);
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(segment_embedding);
  COPY_ATTRIBUTE_FLOAT_TENSOR_TO_GPU(position_embedding);

  ORT_ENFORCE(word_embedding_proto.dims().size() == 2);
  ORT_ENFORCE(position_embedding_proto.dims().size() == 2);
  ORT_ENFORCE(segment_embedding_proto.dims().size() == 2);
  ORT_ENFORCE(word_embedding_proto.dims()[1] == segment_embedding_proto.dims()[1]);
  ORT_ENFORCE(word_embedding_proto.dims()[1] == position_embedding_proto.dims()[1]);

  hidden_size_ = word_embedding_proto.dims()[1];
}

template <typename T>
Status EmbedLayerNorm<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);
  const Tensor* mask = context->Input<Tensor>(2);

  if (input_ids->Shape() != segment_ids->Shape() || input_ids->Shape() != mask->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "All input tensors shall have same shape");
  }

  const auto input_dims = input_ids->Shape().GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 2 dimensions, got ", input_dims.size());
  }

  std::vector<int64_t> out_dims;
  out_dims.reserve(3);
  out_dims.push_back(input_dims[0]);
  out_dims.push_back(input_dims[1]);
  out_dims.push_back(hidden_size_);
  TensorShape output_shape(out_dims);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> mask_index_dims;
  mask_index_dims.push_back(input_dims[0]);
  TensorShape mask_index_shape(mask_index_dims);
  Tensor* mask_index = context->Output(1, mask_index_shape);

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  size_t element_size = sizeof(T);

#ifdef USE_CUDA_FP16
  launchEmbedLayerNormKernel(
      output->template MutableData<T>(),
      mask_index->template MutableData<int32_t>(),
      input_ids->template Data<int32_t>(),
      segment_ids->template Data<int32_t>(),
      mask->template Data<int32_t>(),
      gamma_data_.get(),
      beta_data_.get(),
      word_embedding_data_.get(),
      position_embedding_data_.get(),
      segment_embedding_data_.get(),
      static_cast<int>(hidden_size_),
      batch_size,
      sequence_length,
      element_size);
#endif

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
