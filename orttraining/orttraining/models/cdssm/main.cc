// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/platform/env.h"
#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"

#include "orttraining/models/mnist/mnist_data_provider.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"

#include <condition_variable>
#include <mutex>
#include <tuple>

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               onnxruntime::ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo);
}

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

struct CdssmParameters : public TrainingRunner::Parameters {
  int max_query_length = 2;
  int max_doc_length = 7;
  int input_dim = 3001;
};

const static int NUM_CLASS = 10;
const static vector<int64_t> IMAGE_DIMS = {784};  //{1, 28, 28} for mnist_conv
const static vector<int64_t> LABEL_DIMS = {10};

Status ParseArguments(int argc, char* argv[], CdssmParameters& params) {
  cxxopts::Options options("POC Training", "Main Program to train on MNIST");
  // clang-format off
  options
    .add_options()
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("log_dir", "The directory to write tensorboard events.",
        cxxopts::value<std::string>()->default_value(""))
      ("use_profiler", "Collect runtime profile data during this training run.", cxxopts::value<bool>()->default_value("false"))
      ("num_train_steps", "Number of training steps.", cxxopts::value<int>()->default_value("100"))
      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>()->default_value("2"))
      ("learning_rate", "The initial learning rate for Adam.", cxxopts::value<float>()->default_value("0.01"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.model_name = flags["model_name"].as<std::string>();
    params.lr_params.initial_lr = flags["learning_rate"].as<float>();
    params.num_train_steps = flags["num_train_steps"].as<int>();
    params.batch_size = flags["train_batch_size"].as<int>();

    //auto train_data_dir = flags["train_data_dir"].as<std::string>();
    auto log_dir = flags["log_dir"].as<std::string>();
    //params.train_data_dir.assign(train_data_dir.begin(), train_data_dir.end());
    params.log_dir.assign(log_dir.begin(), log_dir.end());
    params.use_profiler = flags.count("use_profiler") > 0;
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << "\n"
         << options.help() << "\n";
    return Status(ONNXRUNTIME, FAIL, msg);
  }
  return Status::OK();
}

// NOTE: these variables need to be alive when the error_function is called.
int true_count = 0;
float total_loss = 0.0f;

void setup_training_params(CdssmParameters& params) {
  params.model_path = ToPathString(params.model_name) + ORT_TSTR(".onnx");
  params.model_with_loss_func_path = ToPathString(params.model_name) + ORT_TSTR("_with_cost.onnx");
  params.model_with_training_graph_path = ToPathString(params.model_name) + ORT_TSTR("_bw.onnx");
  params.model_actual_running_graph_path = ToPathString(params.model_name) + ORT_TSTR("_bw_running.onnx");
  params.output_dir = ORT_TSTR(".");

  params.fetch_names = {"loss"};

  params.training_optimizer_name = "SGDOptimizer";

  params.weights_to_train = {"_q_conv.weight", "_q_conv.bias", "_q_sem.weight", "_q_sem.bias", "_d_conv.weight", "_d_conv.bias", "_d_sem.weight", "_d_sem.bias"};

  std::shared_ptr<EventWriter> tensorboard;
  if (!params.log_dir.empty() && MPIContext::GetInstance().GetWorldRank() == 0)
    tensorboard = std::make_shared<EventWriter>(params.log_dir);

  params.post_evaluation_callback = [tensorboard](size_t num_samples, size_t step, const std::string /**/) {
    float precision = float(true_count) / num_samples;
    float average_loss = total_loss / float(num_samples);
    if (tensorboard != nullptr) {
      tensorboard->AddScalar("precision", precision, step);
      tensorboard->AddScalar("loss", average_loss, step);
    }
    printf("Step: %zu, #examples: %d, #correct: %d, precision: %0.04f, loss: %0.04f \n\n",
           step,
           static_cast<int>(num_samples),
           true_count,
           precision,
           average_loss);
    true_count = 0;
    total_loss = 0.0f;
  };
}

int main(int argc, char* args[]) {
  // setup logger
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  // setup onnxruntime env
  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(nullptr, env).IsOK());

  // setup training params
  CdssmParameters params;
  RETURN_IF_FAIL(ParseArguments(argc, args, params));
  setup_training_params(params);

  // setup data
  /*auto device_count = MPIContext::GetInstance().GetWorldSize();
  std::vector<string> feeds{"query", "doc"};
  auto trainingData = std::make_shared<DataSet>(feeds);*/

  // setup fake data
  const int batch_size = static_cast<int>(params.batch_size);
  std::vector<std::string> tensor_names = {"query",
                                           "doc"};
  std::vector<TensorShape> tensor_shapes = {{batch_size, params.max_query_length, params.input_dim},
                                            {batch_size, params.max_doc_length, params.input_dim}};
  std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_FLOAT};
  const size_t num_of_perf_samples = params.num_train_steps * params.batch_size;
  auto random_perf_data = std::make_shared<RandomDataSet>(num_of_perf_samples, tensor_names, tensor_shapes, tensor_types);
  auto random_perf_data_loader = onnxruntime::make_unique<SingleDataLoader>(random_perf_data, tensor_names);

  TrainingRunner runner{params, *env};
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run(random_perf_data_loader.get(), random_perf_data_loader.get()));
  
  // start training session
  /*auto training_data_loader = std::make_shared<SingleDataLoader>(trainingData, feeds);
  auto runner = onnxruntime::make_unique<TrainingRunner>(params, *env);
  RETURN_IF_FAIL(runner->Initialize());
  RETURN_IF_FAIL(runner->Run(training_data_loader.get(), nullptr));
  RETURN_IF_FAIL(runner->EndTraining(nullptr));*/
}
