// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets-ml.h"
#include "onnx/defs/operator_sets-training.h"
#ifndef DISABLE_CONTRIB_OPS
#include "core/graph/contrib_ops/contrib_defs.h"
#endif
#ifdef ML_FEATURIZERS
#include "core/graph/featurizers_ops/featurizers_defs.h"
#endif
#ifdef USE_DML
#include "core/graph/dml_ops/dml_defs.h"
#endif

#include "core/platform/env.h"
#include "core/util/thread_utils.h"

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include "core/platform/tracing.h"
#endif

using namespace ::onnxruntime::common;
using namespace ::onnxruntime;
using namespace ONNX_NAMESPACE;

std::once_flag schemaRegistrationOnceFlag;

namespace {
struct ThreadPoolPair {
  onnxruntime::concurrency::ThreadPool* intra_op_thread_pool_ = nullptr;
  onnxruntime::concurrency::ThreadPool* inter_op_thread_pool_ = nullptr;
};
//Singleton
class GlobalThreadPool {
 private:
  //If user forgot to delete the thread pools, let it happen.
  //Because we don't know if it is safe to call the destructor now.
  ThreadPoolPair tp_;
  OrtMutex m_;
  int ref_count_ = 0;
  constexpr GlobalThreadPool() = default;

  //static initialized
  static GlobalThreadPool instance_;

 public:
  static void Release() {
    ThreadPoolPair ToDelete;
    {
      std::lock_guard<OrtMutex> l(instance_.m_);
      if (--instance_.ref_count_ == 0) {
        ToDelete = instance_.tp_;
      }
    }
    delete ToDelete.intra_op_thread_pool_;
    delete ToDelete.inter_op_thread_pool_;
  }

  //Always return a copy of instance_.tp_
  static ThreadPoolPair Get(const OrtThreadingOptions* tp_options) {
    ThreadPoolPair ret;
    {
      std::lock_guard<OrtMutex> l(instance_.m_);
      if (instance_.ref_count_ == 0) {
        OrtThreadPoolParams to = tp_options->intra_op_thread_pool_params;
        if (to.name == nullptr) {
          to.name = ORT_TSTR("intra-op");
        }
        instance_.tp_.intra_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, nullptr).release();
        to = tp_options->inter_op_thread_pool_params;
        if (to.name == nullptr) {
          to.name = ORT_TSTR("inter-op");
        }
        instance_.tp_.inter_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, nullptr).release();
        instance_.ref_count_ = 1;
      } else
        ++instance_.ref_count_;
      ret = instance_.tp_;
    }
    return ret;
  }
};
GlobalThreadPool GlobalThreadPool::instance_;
}  // namespace

OrtEnv::~OrtEnv() {
  if (create_global_thread_pools_)
    GlobalThreadPool::Release();
}
Status OrtEnv::Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                      std::unique_ptr<OrtEnv>& ret,
                      const OrtThreadingOptions* tp_options,
                      bool create_global_thread_pools) NO_EXCEPTION {
  try {
    ret.reset(new OrtEnv());
  } catch (std::exception& ex) {
    return Status(ONNXRUNTIME, FAIL, ex.what());
  }
  auto status = ret->Initialize(std::move(logging_manager), tp_options, create_global_thread_pools);
  return status;
}

Status OrtEnv::Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                          const OrtThreadingOptions* tp_options,
                          bool create_global_thread_pools) {
  auto status = Status::OK();

  logging_manager_ = std::move(logging_manager);

  // create thread pools
  if (create_global_thread_pools) {
    create_global_thread_pools_ = true;
    ThreadPoolPair g = GlobalThreadPool::Get(tp_options);
    intra_op_thread_pool_ = g.intra_op_thread_pool_;
    inter_op_thread_pool_ = g.inter_op_thread_pool_;
  }

  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSFeaturizersDomain, 1, 1);
#ifdef USE_DML
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDmlDomain, 1, 1);
#endif
      // Register contributed schemas.
      // The corresponding kernels are registered inside the appropriate execution provider.
#ifndef DISABLE_CONTRIB_OPS
      contrib::RegisterContribSchemas();
#endif
#ifdef ML_FEATURIZERS
      featurizers::RegisterMSFeaturizersSchemas();
#endif
#ifdef USE_DML
      dml::RegisterDmlSchemas();
#endif
      RegisterOnnxOperatorSetSchema();
      RegisterOnnxMLOperatorSetSchema();
      RegisterOnnxTrainingOperatorSetSchema();
    });

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx
    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    // fire off startup telemetry (this call is idempotent)
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogProcessInfo();
  } catch (std::exception& ex) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what()};
  } catch (...) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION};
  }

  return status;
}
