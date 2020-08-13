// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class BiasDropoutSoftmaxFusion
Fuse Dropout(Softmax(Input + Bias))
*/
class BiasDropoutSoftmaxFusion : public GraphTransformer {
 public:
  BiasDropoutSoftmaxFusion(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("BiasDropoutSoftmaxFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
