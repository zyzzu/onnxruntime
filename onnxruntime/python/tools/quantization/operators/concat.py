import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain, find_by_name
from onnx import onnx_pb as onnx_proto
import numpy


class QConcat(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # only try to quantize when given quantization parameters for it
        data_found, output_scale_name, output_zp_name, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            super().quantize()
            return

        # mapping of fp32 model
        output_name_to_node = self.quantizer.model.output_name_to_node()
        output_zp = find_by_name(output_zp_name, self.quantizer.model.initializer())
        output_scale = find_by_name(output_scale_name, self.quantizer.model.initializer())
        if not output_zp or not output_scale:
            super().quantize()
            return
        output_zp = numpy.asscalar(onnx.numpy_helper.to_array(output_zp))
        output_scale = numpy.asscalar(onnx.numpy_helper.to_array(output_scale))

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))

        nodes = []
        quantized_inputs = []
        for i, input_name in enumerate(node.input):
            quantized_input_names, zero_point_names, scale_names, nodes_single_input = self.quantizer.quantize_inputs(node, [i])
            nodes.extend(nodes_single_input)
            input_zp = numpy.asscalar(onnx.numpy_helper.to_array(
                find_by_name(zero_point_names[0], self.quantizer.model.initializer())))
            input_scale = numpy.asscalar(onnx.numpy_helper.to_array(
                find_by_name(scale_names[0], self.quantizer.model.initializer())))

            if input_zp == output_zp and input_scale == output_scale:
                quantized_inputs.append(quantized_input_names[0])
            else:
                intermedia_dequantize_name = quantized_input_names[0] + "_dequantize"
                intermedia_requantize_name = quantized_input_names[0] + "_requantized"
                dqlinear_inputs = [quantized_input_names[0], scale_names[0], zero_point_names[0]]
                dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [intermedia_dequantize_name],
                                                        node.name + "_dequantize_intput" + str(i))
                qlinear_node = onnx.helper.make_node("QuantizeLinear", [intermedia_dequantize_name, output_scale_name, output_zp_name],
                                                     [intermedia_requantize_name],  node.name + "_requantize_intput" + str(i))
                quantized_inputs.append(intermedia_requantize_name)
                nodes.extend([dequantize_node, qlinear_node])

        quantized_output_name = node.output[0] + "quantized"
        q_output = QuantizedValue(node.output[0], quantized_output_name, output_scale_name,
                                  output_zp_name, QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        quantized_node = onnx.helper.make_node(
            node.op_type, quantized_inputs, [quantized_output_name], node.name + "_quantized" if node.name else "", **kwargs)
        nodes.append(quantized_node)

        self.quantizer.new_nodes += nodes
