import onnx
import shutil
import sys

# nchwc_domain = 'kMSNchwcDomain'
# optimizer_ops = set(['Attention', 'Gelu', 'BiasGelu', 'FastGelu', 'FusedConv', 'FusedGemm', 'DynamicQuantizeMatMul',
#                      'EmbedLayerNormalization', 'LayerNormalization', 'SkipLayerNormalization', 'TransposeMatMul',
#                      'MemcpyFromHost', 'MemcpyToHost'])

# q10n_ops = set(['MatMulInteger16', 'DequantizeLinear', 'QuantizeLinear', 'QLinearLeakyRelu',
#                 'QAttention', 'DynamicQuantizeMatMul'])


def get_model_info(model_path, domain_opset_ops):
    '''
    Get info on the opset and ops used in the model
    :param model_path: Path to model
    :param domain_opset_ops: Add information to this map of {domain:{opset:[ops]}}
    '''

    m = onnx.load_model(model_path)
    target_opsets = {}
    ops = {}
    for entry in m.opset_import:
        domain = entry.domain if len(entry.domain) > 0 else 'ai.onnx'
        target_opsets[domain] = entry.version
        ops[domain] = set()
        # add keys to domain_opset_ops if not present
        if domain not in domain_opset_ops:
            domain_opset_ops[domain] = {entry.version: set()}
        else:
            if entry.version not in domain_opset_ops[domain]:
                domain_opset_ops[domain][entry.version] = set()

    def process_nodes(graph):
        for n in graph.node:
            d = n.domain if len(n.domain) > 0 else 'ai.onnx'  # empty == onnx
            ops[d].add(n.op_type)
            domain_opset_ops[d][target_opsets[d]].add(n.op_type)

            for attr in n.attribute:
                if attr.HasField('g'):
                    process_nodes(attr.g)

    process_nodes(m.graph)

    return domain_opset_ops


# todo: read from core/graph/constants.h
domain_map = {'kOnnxDomain': 'ai.onnx',
              'kMLDomain': 'ai.onnx.ml',
              'kMSDomain': 'com.microsoft',
              'kMSFeaturizersDomain': 'com.microsoft.mlfeaturizers',
              'kMSNchwcDomain': 'com.microsoft.nchwc'}


def process_block(domain_opset_ops, is_contrib_ops, block, is_typed, is_versioned, orig_lines, out):

    extracted = block[block.find('(') + 1:block.find(')')]
    pieces = [x.strip() for x in extracted.split(',')]
    if pieces[1] not in domain_map:
        print(f'Invalid block was extracted. Domain info was incorrect: {extracted}')
        sys.exit(-1)

    domain = domain_map[pieces[1]]
    start = pieces[2]
    # type = ''

    if not is_versioned and not is_typed:
        end = 999
        op = pieces[3]
    elif is_versioned and not is_typed:
        end = pieces[3]
        op = pieces[4]
    elif is_typed and not is_versioned:
        end = 999
        # type = pieces[3]
        op = pieces[4]
    else:
        assert(is_typed and is_versioned)
        end = pieces[3]
        # type = pieces[4]
        op = pieces[5]

    # check if enabled
    enabled = False

    # if domain wasn't found in the model opset imports, it is disabled by default
    if domain in domain_opset_ops:
        for opset, ops in domain_opset_ops[domain].items():
            if int(start) <= opset and int(end) >= opset and op in ops:
                enabled = True
                break

            # todo: if the build is going to have optimizers that run during initialization we need to whitelist
            # any ops that may be needed by the optimizers

    # print(f'{enabled} domain={domain} start={start} end={end} type={type} op={op}')

    for line in orig_lines:
        if not enabled:
            out.write('// ')
        out.write(line)

    # add closing brace if we just commented out the last line in the kernel registry table
    if not enabled and block.endswith('>};'):
        out.write('  };')


# Process the execution provider looking for these 4 ONNX_OPERATOR_* macros which are used in 2 different ways
#
# class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, Transpose);
# class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
# class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 8, float, Expand);
# class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);

# BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, Cos)>,
# BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 7, double, Sin)>,
# BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
#                                                                   Hardmax)>,
# BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, 10,
#                                                                         float, LogSoftmax)>,

def process_file(domain_opset_ops, input_filename, is_contrib_ops=False):
    block = ''
    in_forward_dec = False
    in_kci = False
    is_typed = False
    is_versioned = False

    orig_lines = []
    new_filename = input_filename + '.new'

    # read from copy, write to original
    with open(input_filename) as f, open(new_filename, 'w') as out:
        orig_line = f.readline()
        while orig_line:
            orig_lines.append(orig_line)
            line = orig_line.strip()

            if line.startswith('//'):
                if(in_forward_dec or in_kci):
                    print("Unexpected commented out line in block.")
                    print(orig_line)
                    sys.exit(-1)
            else:
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
                    # skip dummy entry
                    if 'BuildKernelCreateInfo<void>' not in line:
                        in_forward_dec = False
                        in_kci = True

            if not in_forward_dec and not in_kci:
                out.write(orig_line)
                orig_lines.clear()
            else:
                block += line

            # check for end of line to do a match/extract on the full info
            if (in_forward_dec and line.endswith(';')) or (in_kci and (line.endswith('>,') or line.endswith('>};'))):
                process_block(domain_opset_ops, is_contrib_ops, block, is_typed, is_versioned, orig_lines, out)
                orig_lines.clear()
                block = ''
                in_forward_dec = False
                in_kci = False

            orig_line = f.readline()

        for line in orig_lines:
            out.write(line)

    # finished processing successfully. swap files arount
    input_copy_filename = input_filename + '.orig'
    shutil.copy(input_filename, input_copy_filename)
    shutil.move(new_filename, input_filename)


if __name__ == "__main__":
    # model_paths = \
    #     [
    #      r'D:\src\github\ORT test models\20190729\opset10\yolov3\yolov3.optimized.onnx',
    #      r'D:\src\github\ORT test models\20190729\opset8\tf_mobilenet_v2_1.4_224\model.optimized.onnx',
    #      r'D:\src\github\ORT test models\20190729\opset10\mlperf_ssd_mobilenet_300\ssd_mobilenet_v1_coco_2018_01_28.optimized.onnx'  # noqa
    #     ]

    model_paths = [
        # r'D:\temp\law_LM\test1\law_LM.optimized.onnx',
        r'C:\Users\scmckay\Desktop\share_with_frank\testing\fluency_v2_opt.quantized.optimized.onnx'
        # r'C:\Users\scmckay\Desktop\OnnxFootprint\quantized.optimized_level2.onnx',
        # r'D:\src\github\ort.deserialize\build\Windows\Debug\Debug\testdata\ort_github_issue_4031.onnx'
    ]

    # {domain: {opset:[list of ops used]}}
    # map of domain to a list of ops used in each opset enabled for that domain
    # as we support using multiple models as input there may be multiple opsets needed for a domain
    domain_opset_ops = {}
    for model_path in model_paths:
        get_model_info(model_path, domain_opset_ops)

    debug = True
    if debug:
        total_ops = 0
        for domain in sorted(domain_opset_ops.keys()):
            opset_to_ops = domain_opset_ops[domain]
            ops_in_domain = sum([len(x) for x in opset_to_ops.values()])
            if ops_in_domain > 0:
                print(f"Domain: {domain}")
                total_ops += ops_in_domain
                for opset, ops in opset_to_ops.items():
                    print(f"\t{opset}:{','.join(sorted(ops))}")

        print(f"Total ops:{total_ops}")

    process_file(domain_opset_ops,
                 r'D:\src\github\ort.deserialize\onnxruntime\core\providers\cpu\cpu_execution_provider.cc',
                 False)

    process_file(domain_opset_ops,
                 r'D:\src\github\ort.deserialize\onnxruntime\contrib_ops\cpu\cpu_contrib_kernels.cc',
                 True)
