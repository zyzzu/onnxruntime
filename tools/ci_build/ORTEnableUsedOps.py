import onnx

nchwc_domain = 'kMSNchwcDomain'
optimizer_ops = set(['Attention', 'Gelu', 'BiasGelu', 'FastGelu', 'FusedConv', 'FusedGemm', 'DynamicQuantizeMatMul',
                     'EmbedLayerNormalization', 'LayerNormalization', 'SkipLayerNormalization', 'TransposeMatMul',
                     'MemcpyFromHost', 'MemcpyToHost'])

# q10n_ops = set(['MatMulInteger16', 'DequantizeLinear', 'QuantizeLinear', 'QLinearLeakyRelu',
#                 'QAttention', 'DynamicQuantizeMatMul'])

def get_model_info(model_path):

    m = onnx.load_model(model_path)
    target_opsets = {}
    ops = {}
    for entry in m.opset_import:
        target_opsets[entry.domain] = entry.version
        ops[entry.domain] = set()

    for n in m.graph.node:
        d = n.domain if len(n.domain) > 0 else 'ai.onnx'  # empty == onnx
        ops[d].add(n.op_type)

    ops1 = ops
    ops = {}
    for k in ops1.keys():
        ops[k] = set()

    def process_nodes(graph):
        for n in graph.node:
            d = n.domain if len(n.domain) > 0 else 'ai.onnx'  # empty == onnx
            ops[d].add(n.op_type)

            for attr in n.attribute:
                if attr.HasField('g'):
                    process_nodes(attr.g)

    process_nodes(m.graph)

    return target_opsets, ops


# todo: read from core/graph/constants.h
domain_map = {'kOnnxDomain': 'ai.onnx',
              'kMLDomain': 'ai.onnx.ml',
              'kMSDomain': 'com.microsoft',
              'kMSFeaturizersDomain': 'com.microsoft.mlfeaturizers',
              'kMSNchwcDomain': 'com.microsoft.nchwc'}


def process_block(target_opsets, enabled_ops, is_contrib_ops, block, is_typed, is_versioned, orig_lines, out):

    extracted = block[block.find('(') + 1:block.find(')')]
    pieces = [x.strip() for x in extracted.split(',')]
    domain = domain_map[pieces[1]]
    start = pieces[2]
    type = ''

    if not is_versioned and not is_typed:
        end = 999
        op = pieces[3]
    elif is_versioned and not is_typed:
        end = pieces[3]
        op = pieces[4]
    elif is_typed and not is_versioned:
        end = 999
        type = pieces[3]
        op = pieces[4]
    else:
        assert(is_typed and is_versioned)
        end = pieces[3]
        type = pieces[4]
        op = pieces[5]

    print(f'domain={domain} start={start} end={end} type={type} op={op}')

    # check if enabled
    if domain in target_opsets:
        target_opset = target_opsets[domain]
    else:
        target_opset = 0

    is_used_in_model = domain in enabled_ops and op in enabled_ops[domain]
    if is_contrib_ops:
        # keep the op if it's used, it's needed by an optimizer (including nchwc) or it's a quantization op
        # todo: may not need the check on q10n
        # Currently the contrib ops are all opset 1 so not checking against target_opset
        enabled = is_used_in_model or op in optimizer_ops or domain == domain_map[nchwc_domain] #or op in q10n_ops
    else:
        enabled = is_used_in_model and int(start) <= target_opset and int(end) >= target_opset

    for l in orig_lines:
        if not enabled:
            out.write('// ')
        out.write(l)


# Process the execution provider looking for these 4 ONNX_OPERATOR_* macros which are used in 2 different ways
#
# class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, Transpose);
# class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
# class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, float, Expand);
# class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);

# BuildKernelCreateInfo < ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Cos) >,
# BuildKernelCreateInfo < ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, double, Sin) >,
# BuildKernelCreateInfo < ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
#                                                                   Hardmax) >,
# BuildKernelCreateInfo < ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
#                                                                         float, LogSoftmax) >,

def process_file(target_opset, ops, input_filename, is_contrib_ops=False):
    block = ''
    in_forward_dec = False
    in_kci = False
    is_typed = False
    is_versioned = False

    orig_lines = []

    with open(input_filename) as f, \
         open(input_filename + '.reduced.txt','w') as out:
        orig_line = f.readline()
        while orig_line:
            orig_lines.append(orig_line)
            line = orig_line.strip()

            if 'ONNX_OPERATOR_KERNEL_CLASS_NAME' in line:
                in_forward_dec = True
                is_typed = False
                is_versioned = False
                in_kci = False
            elif 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME' in line:
                in_forward_dec = True
                is_typed = True
                is_versioned = False
                in_kci = False
            elif 'ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME' in line:
                in_forward_dec = True
                is_typed = False
                is_versioned = True
                in_kci = False
            elif 'ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME' in line:
                in_forward_dec = True
                is_typed = True
                is_versioned = True
                in_kci = False

            if 'BuildKernelCreateInfo<' in line:
                in_forward_dec = False
                in_kci = True

            if not in_forward_dec and not in_kci:
                out.write(orig_line)
                orig_lines.clear()
            else:
                block += line

            # check for end of line to do a match/extract on the full info
            if (in_forward_dec and line.endswith(';')) or (in_kci and line.endswith('>,')):
                process_block(target_opset, ops, is_contrib_ops, block, is_typed, is_versioned, orig_lines, out)
                orig_lines.clear()
                block = ''
                in_forward_dec = False
                in_kci = False

            orig_line = f.readline()

        for line in orig_lines:
            out.write(line)

if __name__ == "__main__":
    model_path = r'C:\Users\scmckay\Desktop\OnnxFootprint\quantized.optimized_level2.onnx'
    target_opset, ops = get_model_info(model_path)
    process_file(target_opset, ops,
                 r'D:\src\github\ort.vs19.perf.master\onnxruntime\core\providers\cpu\cpu_execution_provider.cc',
                 False)
    # process_file(target_opset, ops,
    #              r'D:\src\github\ort.vs19.perf.master\onnxruntime\contrib_ops\cpu\cpu_contrib_kernels.cc',
    #              True)
