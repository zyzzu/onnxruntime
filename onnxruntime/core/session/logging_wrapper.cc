// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//this file contains implementations of the C API

#include <cassert>

#include "logging_wrapper.h"
#include "core/session/ort_apis.h"
#include "core/session/environment.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/logging.h"

using namespace onnxruntime;
using namespace onnxruntime::logging;

LoggingWrapper::LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param)
    : logging_function_(logging_function), logger_param_(logger_param) {
}

void LoggingWrapper::SendImpl(const onnxruntime::logging::Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                              const onnxruntime::logging::Capture& message) {
  std::string s = message.Location().ToString();
  logging_function_(logger_param_, static_cast<OrtLoggingLevel>(message.Severity()), message.Category(),
                    logger_id.c_str(), s.c_str(), message.Message().c_str());
}