// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/optimizer_builder.h"

namespace onnxruntime {
namespace training {

class SGDOptimizerBuilder final : public OptimizerBuilder {
 public:
  SGDOptimizerBuilder() : OptimizerBuilder(OpDef{"SGDOptimizer", kMSDomain, 1}) {}

  virtual Status Build(
      const OptimizerBuilderConfig& config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs) const override;
};

}  // namespace training
}  // namespace onnxruntime
