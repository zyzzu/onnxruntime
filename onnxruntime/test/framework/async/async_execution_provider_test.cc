// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/inference_session.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"

#include "gtest/gtest.h"
#include "../test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
class AsyncFuseAdd : public OpKernel {
 public:
  explicit AsyncFuseAdd(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto X = context->Input<Tensor>(0);
    auto Y = context->Input<Tensor>(1);
    auto Z = context->Input<Tensor>(2);
    auto& shape = X->Shape();
    auto M = context->Output(0, shape)->template MutableData<float>();
    for (int i = 0; i < shape.Size(); ++i) {
      *(M + i) = *(X->template Data<float>() + i) + *(Y->template Data<float>() + i) + *(Z->template Data<float>() + i);
    }
    return Status::OK();
  }
};

constexpr const char* kAsyncFuseTest = "AsyncFuseTest";
constexpr const char* kAsyncFuseExecutionProvider = "AsyncFuseExecutionProvider";
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kFuseExecutionProvider, kFuseTest, 1, FuseAdd);
ONNX_OPERATOR_KERNEL_EX(AsyncFuseAdd,
                        kAsyncFuseTest,
                        1,
                        kAsyncFuseExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        AsyncFuseAdd);

class AsyncFuseExecutionProvider : public IExecutionProvider {
 public:
  explicit AsyncFuseExecutionProvider() : IExecutionProvider{kAsyncFuseExecutionProvider} {
    AllocatorCreationInfo device_info{
        [](int) {
          return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo("AsyncFuse", OrtAllocatorType::OrtDeviceAllocator));
        }};
    InsertAllocator(device_info.device_alloc_factory(0));
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override {
    // Fuse two add into one.
    std::vector<std::unique_ptr<ComputeCapability>> result;
    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    for (auto& node : graph.Nodes()) {
      sub_graph->nodes.push_back(node.Index());
    }
    auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = "FuseAdd";
    meta_def->domain = "FuseTest";
    meta_def->inputs = {"X", "Y", "Z"};
    meta_def->outputs = {"M"};
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    sub_graph->SetMetaDef(std::move(meta_def));
    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
    return result;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    static std::shared_ptr<KernelRegistry> kernel_registry;
    if (kernel_registry == nullptr) {
      kernel_registry = std::make_shared<KernelRegistry>();
      ORT_ENFORCE(kernel_registry->Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAsyncFuseExecutionProvider, kAsyncFuseTest, 1, AsyncFuseAdd)>()).IsOK());
    }
    return kernel_registry;
  }
};

namespace test {
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<float>& expected_values);

TEST(ExecutionProviderTest, AsyncExecutionProviderTest) {
  onnxruntime::Model model("graph_1", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "async_execution_provider_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "ExecutionProviderTest.AsyncExecutionProviderTest";
  InferenceSession session_object{so, GetEnvironment()};
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = onnxruntime::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);

  InferenceSession session_object_2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object_2.RegisterExecutionProvider(std::move(testCPUExecutionProvider)));
  ASSERT_STATUS_OK(
      session_object_2.RegisterExecutionProvider(onnxruntime::make_unique<::onnxruntime::AsyncFuseExecutionProvider>()));
  status = session_object_2.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object_2.Initialize();
  ASSERT_TRUE(status.IsOK());
  status = session_object_2.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

}  // namespace test
}  // namespace onnxruntime
