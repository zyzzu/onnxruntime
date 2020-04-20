// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <string>
#include "core/session/onnxruntime_c_api.h"
#include "core/common/logging/isink.h"

//Create an ISink from a C style logging function
class LoggingWrapper : public onnxruntime::logging::ISink {
 public:
  LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param);

  void SendImpl(const onnxruntime::logging::Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                const onnxruntime::logging::Capture& message) override;

 private:
  OrtLoggingFunction logging_function_;
  void* logger_param_;
};
