# CUDA Execution Provider

The CUDA execution provider in the ONNX Runtime makes use of NVIDIA's [CUDA](https://developer.nvidia.com/cuda) parallel computing platform and programming model to accelerate ONNX model in their family of GPUs. Microsoft and NVIDIA worked closely to integrate the CUDA execution provider with ONNX Runtime.

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#cuda). 

The CUDA execution provider for ONNX Runtime is built and tested with CUDA 11.0.1.

## Using the CUDA execution provider
### C/C++
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, device_id));
Ort::Session session(env, model_path, sf);
```
The C API details are [here](../C_API.md#c-api).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)

When/if using [onnxruntime_perf_test](../../onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e cuda` 