// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <memory>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/platform/threadpool.h"
#include "core/common/logging/logging.h"

struct OrtThreadingOptions;

//Not a singleton
struct OrtEnv {
 public:
  /**
     Create and initialize the runtime OrtEnv.
    @param logging manager instance that will enable per session logger output using
    session_options.session_logid as the logger id in messages.
    If nullptr, the default LoggingManager MUST have been created previously as it will be used
    for logging. This will use the default logger id in messages.
    See core/common/logging/logging.h for details, and how LoggingManager::DefaultLogger works.
    @param tp_options optional set of parameters controlling the number of intra and inter op threads for the global
    threadpools.
    @param create_global_thread_pools determine if this function will create the global threadpools or not.
  */
  static onnxruntime::common::Status Create(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager,
                                            std::unique_ptr<OrtEnv>& OrtEnv,
                                            const OrtThreadingOptions* tp_options = nullptr,
                                            bool create_global_thread_pools = false) NO_EXCEPTION;

  onnxruntime::logging::LoggingManager* GetLoggingManager() const {
    return logging_manager_.get();
  }

  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager) {
    logging_manager_ = std::move(logging_manager);
  }

  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPool() const {
    return intra_op_thread_pool_;
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPool() const {
    return inter_op_thread_pool_;
  }

  bool EnvCreatedWithGlobalThreadPools() const {
    return create_global_thread_pools_;
  }
  ~OrtEnv();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtEnv);

 private:
  constexpr OrtEnv() = default;
  onnxruntime::common::Status Initialize(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager,
                                         const OrtThreadingOptions* tp_options = nullptr,
                                         bool create_global_thread_pools = false);

  std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager_;
  onnxruntime::concurrency::ThreadPool* intra_op_thread_pool_ = nullptr;
  onnxruntime::concurrency::ThreadPool* inter_op_thread_pool_ = nullptr;
  bool create_global_thread_pools_{false};
};
