import os
import sys

# todo: read from core/graph/constants.h
domain_map = {'kOnnxDomain': 'ai.onnx',
              'kMLDomain': 'ai.onnx.ml',
              'kMSDomain': 'com.microsoft',
              'kMSFeaturizersDomain': 'com.microsoft.mlfeaturizers',
              'kMSNchwcDomain': 'com.microsoft.nchwc'}


def process_block(block, is_typed, is_versioned):

    extracted = block[block.find('(') + 1:block.find(')')]
    pieces = [x.strip() for x in extracted.split(',')]
    if pieces[1] not in domain_map:
        print(f'Invalid block was extracted. Domain info was incorrect: {extracted}')
        sys.exit(-1)

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

    return domain, op, int(start), int(end), type


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

def process_file(domain_opset_ops, input_filename):
    block = ''
    in_forward_dec = False
    in_kci = False
    is_typed = False
    is_versioned = False

    orig_lines = []

    # read from copy, write to original
    with open(input_filename) as f:
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
                orig_lines.clear()
            else:
                block += line

            # check for end of line to do a match/extract on the full info
            if (in_forward_dec and line.endswith(';')) or (in_kci and (line.endswith('>,') or line.endswith('>};'))):

                if in_kci:
                    domain, op, start, end, type = process_block(block, is_typed, is_versioned)

                    if domain not in domain_opset_ops:
                        domain_opset_ops[domain] = {}

                    if op not in domain_opset_ops[domain]:
                        domain_opset_ops[domain][op] = []

                    domain_opset_ops[domain][op].append((start, end, type))

                orig_lines.clear()
                block = ''
                in_forward_dec = False
                in_kci = False

            orig_line = f.readline()


def validate_registrations(domain_op_registrations):
    """Validate registrations.
    Input is {domain:{op:[(start,end,type),...]

    Printed output for each op registration, sorted by domain, op and type.
    Invalid entries have a message with 'INVALID' prior to the bad registration.
    Suspicious entries have a message with 'CHECK' prior to all registrations for the op.
    """

    for domain in sorted(domain_op_registrations.keys()):
        op_to_registrations = domain_op_registrations[domain]
        for op in sorted(op_to_registrations.keys()):
            regos = op_to_registrations[op]
            print(f'{domain}:{op}')

            # sort on start version to check the number of types per version didn't change
            s = sorted(regos, key=lambda x: (x[0]))

            # count types per start version
            types_per_version = {}
            for entry in s:
                start, end, type = entry
                if start not in types_per_version:
                    types_per_version[start] = set()

                if type == '':
                    type = 'all'

                types_per_version[start].add(type)

            num_versions = len(types_per_version)
            if num_versions > 1:
                sorted_versions = sorted(types_per_version.keys())
                for v in range(num_versions - 1):
                    old_ver = sorted_versions[v]
                    new_ver = sorted_versions[v + 1]
                    old_types = types_per_version[old_ver]
                    new_types = types_per_version[new_ver]

                    # old_type must be in new_types unless new_types == 'all'
                    if 'all' in new_types:
                        assert(len(new_types) == 1)
                        continue

                    for type in old_types:
                        if type not in new_types:
                            print(f'CHECK: Support for {type} was in {old_ver} but not {new_ver}')

            # sort on type and start version to check for version overlap
            s = sorted(regos, key=lambda x: (x[2], x[0]))
            prev = None
            for entry in s:
                start, end, type = entry
                if prev:
                    prev_start, prev_end, prev_type = prev
                    # validate no overlap in end version
                    if prev_type == type:
                        if prev_end != start - 1:
                            print("INVALID. Overlap between end and start versions. ")

                print(f'{start}\t{end}\t{type}')
                prev = entry


if __name__ == "__main__":

    repo = r'D:\src\github\ort.vs19.1'
    registrations = [
        r'onnxruntime\core\providers\cpu\cpu_execution_provider.cc',
        r'onnxruntime\core\providers\cuda\cuda_execution_provider.cc',
        r'onnxruntime\contrib_ops\cpu\cpu_contrib_kernels.cc'
    ]

    for file in registrations:
        # {domain: {op:[(start,end,type),...]}}
        domain_op_registrations = {}

        filename = os.path.join(repo, file)
        process_file(domain_op_registrations, filename)

        print(filename)
        validate_registrations(domain_op_registrations)
        print('----------------------------------')
