import onnxruntime as ort
import os
# import sys
# sys.path.append(r'D:\src\github\ort.serialize\tools\python')
from ort_test_dir_utils import run_test_dir

test_models = {
    r'D:\src\github\ORT test models\20190729\opset10\yolov3\yolov3.onnx': 'yolov3.ort',
    r'D:\src\github\ORT test models\20190729\opset8\tf_mobilenet_v2_1.4_224\model.onnx': 'tf_mobilenet_v2_1.4_224.ort',
    r'D:\src\github\ORT test models\20190729\opset10\mlperf_ssd_mobilenet_300\ssd_mobilenet_v1_coco_2018_01_28.onnx':
        'mlperf_ssd_mobilenet_300.ort',
    # r'C:\Users\scmckay\Desktop\TFNet\singleframe_optimized_hidden_layers.onnx' : 'tfnet.ort',
    # r'C:\Users\scmckay\Desktop\OnnxFootprint\quantized.onnx' : 'bert_nlu.ort',
}


def run(create=None):

    if create == 'ORT':
        so = ort.SessionOptions()
        so.serialized_model_format = ort.capi.onnxruntime_pybind11_state.SerializationFormat.ORT
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    if create == 'ONNX':
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    for model, target in test_models.items():
        model_dir = os.path.split(model)[0]
        target_path = os.path.join(model_dir, target)

        if create:
            if create == 'ONNX':
                target_path = model.replace('.onnx', '.optimized.onnx')

            so.optimized_model_filepath = target_path

            # so.intra_op_num_threads = num_threads
            print(f"Converting {model}")
            _ = ort.InferenceSession(model, sess_options=so)

            orig_size = os.path.getsize(model)
            new_size = os.path.getsize(target_path)
            print(f"Serialized {model} to {target_path}. Sizes: orig={orig_size} "
                  f"new={new_size} diff={new_size/orig_size:.4f}")

        if not create and not os.path.exists(target_path):
            print(f"Missing serialized ORT format model for {model}")
        else:
            run_test_dir(target_path)
            # os.remove(target_path)

        # sess2 = ort.InferenceSession(target_path, deserialize=True)
        # for i in sess2.get_inputs():
        #    print(i.name)


# run('ONNX')
# run('ORT')
run()
