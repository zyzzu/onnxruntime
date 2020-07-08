// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

#include "flatbuffers/flexbuffers.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {

// InferenceSession wrapper to expose loaded graph.
class InferenceSessionGetGraphWrapper : public InferenceSession {
 public:
  explicit InferenceSessionGetGraphWrapper(const SessionOptions& session_options,
                                           const Environment& env) : InferenceSession(session_options, env) {
  }

  const Graph& GetGraph() {
    return model_->MainGraph();
  }
};

namespace test {
// test that we can deserialize and run a model
TEST(OrtModelOnlyTests, DeserializeToFlexBuffer) {
  const auto output_file = ORT_TSTR("ort_github_issue_4031.onnx.ort");
  SessionOptions so;
  so.session_logid = "DeserializeToFlexBuffer";
  so.optimized_model_filepath = output_file;
  so.optimized_model_format = ORT_INTERNAL_FORMAT;

  InferenceSessionGetGraphWrapper session_object2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.Deserialize(output_file));

  const auto& graph2 = session_object2.GetGraph();
  std::cout << graph2.MaxNodeIndex() << " is max node index\n";

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {123.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"state_var_out"};
  std::vector<OrtValue> fetches;
  std::vector<OrtValue> fetches2;

  ASSERT_STATUS_OK(session_object2.Run(feeds, output_names, &fetches2));

  const auto& output2 = fetches2[0].Get<Tensor>();
  ASSERT_TRUE(output2.Shape().Size() == 1);
  ASSERT_TRUE(output2.Data<float>()[0] == 125.f);
}
}  // namespace test
}  // namespace onnxruntime
