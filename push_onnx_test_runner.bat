D:\Android\platform-tools\adb.exe push D:\src\github\ort.android\build\Windows\RelWithDebInfo\onnx_test_runner /data/local/tmp
D:\Android\platform-tools\adb.exe push D:\src\github\ort.android\build\Windows\RelWithDebInfo\testdata\transform\gemm_activation_fusion /data/local/tmp
D:\Android\platform-tools\adb.exe shell chmod +x /data/local/tmp/onnx_test_runner
D:\Android\platform-tools\adb.exe shell 'cd /data/local/tmp && ./onnx_test_runner gemm_activation_fusion'