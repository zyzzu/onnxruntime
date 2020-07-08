// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_MODEL_FORMAT_ONLY)
namespace onnxruntime {

common::Status CreateCustomRegistry(const std::vector<OrtCustomOpDomain*>& op_domains, std::shared_ptr<CustomRegistry>& output);

}
#endif
