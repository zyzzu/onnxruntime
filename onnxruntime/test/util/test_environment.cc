// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/test_environment.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"

using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

const OrtEnv& GetEnvironment() {
  return **ort_env;
}

::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  OrtEnv& e = **ort_env;
  return *e.GetLoggingManager();
}

}  // namespace test
}  // namespace onnxruntime
