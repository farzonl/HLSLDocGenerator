from enum import Enum
import pypandoc
from bs4 import BeautifulSoup
import os
import sys
import subprocess
import pathlib
import re
from cli import parser
import shutil
import csv

# pylint: disable=wrong-import-position
sys.path.append(os.path.join(os.getcwd(), "DirectXShaderCompiler/utils/hct"))
from DirectXShaderCompiler.utils.hct.hctdb_instrhelp import get_db_dxil
from DirectXShaderCompiler.utils.hct.hctdb_instrhelp import get_db_hlsl
# pylint: enable=wrong-import-position

db_dxil = get_db_dxil()
db_hlsl = get_db_hlsl()
full_md_filepath = os.path.join(
    pathlib.Path().resolve(),
    'win32/desktop-src/direct3dhlsl/dx-graphics-hlsl-intrinsic-functions.md')

shader_semantic_filepath = os.path.join(
    pathlib.Path().resolve(),
    'win32/desktop-src/direct3dhlsl/dx-graphics-hlsl-semantics.md')

pixel_intrinsics = [
    'GetRenderTargetSampleCount',
    'GetRenderTargetSamplePosition',
    'clip',
    'GetAttributeAtVertex',
    'CheckAccessFullyMapped',
    'EvaluateAttributeAtSample',
    'EvaluateAttributeCentroid',
    'EvaluateAttributeSnapped']

pixel_intrinsics_special = [
    'EvaluateAttributeAtSample',
    'EvaluateAttributeCentroid',
    'EvaluateAttributeSnapped']

def gen_deprecated_intrinsics():
    intrinsic_def_path = 'DirectXShaderCompiler/lib/HLSL/HLOperationLower.cpp' 
    pattern = r'IntrinsicOp::IOP_(\w+),\s*EmptyLower'
    with open(intrinsic_def_path, 'r') as int_file:
        int_def = int_file.read()
        matches = re.findall(pattern, int_def)
        return matches


any_hit_intrinsics = ['IgnoreHit', 'AcceptHitAndEndSearch']

node_intrinsics = ['GetRemainingRecursionLevels']

hidden_intrinsics = ['CreateResourceFromHeap', 'AllocateRayQuery']

mesh_intrinsics = ['SetMeshOutputCounts']

amplification_intrinsics = ['DispatchMesh']

hull_intrinsics = [
    'Process2DQuadTessFactorsAvg',
    'Process2DQuadTessFactorsMax',
    'Process2DQuadTessFactorsMin',
    'ProcessIsolineTessFactors',
    'ProcessQuadTessFactorsAvg',
    'ProcessQuadTessFactorsMax',
    'ProcessQuadTessFactorsMin',
    'ProcessTriTessFactorsAvg',
    'ProcessTriTessFactorsMax',
    'ProcessTriTessFactorsMin']

const_intrinsitcs = ['Barrier']

raygeneration_intrinsics = ['TraceRay']

intrinsic_to_spirv_map = { 'printf' : r'NonSemantic\.DebugPrintf', 'D3DCOLORtoUBYTE4': 'OpConvertFToS',
                           'AllMemoryBarrier': 'OpMemoryBarrier', 'countbits': 'OpBitCount',
                           'AllMemoryBarrierWithGroupSync': 'OpControlBarrier', 'fmod': 'OpFRem',
                           'IsHelperLane': 'OpIsHelperInvocationEXT', 'DeviceMemoryBarrierWithGroupSync': 'OpControlBarrier',
                           'mul' : 'OpFMul', 'mul_1' : 'OpVectorTimesScalar', 'mul_2' : 'OpMatrixTimesScalar',
                           'mul_3' : 'OpVectorTimesScalar', 'mul_4' : 'OpDot', 'mul_5' : 'OpMatrixTimesVector',
                           'mul_6' : 'OpMatrixTimesScalar', 'mul_7' : 'OpVectorTimesMatrix', 
                           'mul_8' : 'OpMatrixTimesMatrix', 'ddx' : 'OpDPdx', 'ddx_coarse': 'OpDPdxCoarse',
                           'ddx_fine': 'OpDPdxFine', 'ddy' : 'OpDPdy', 'ddy_coarse': 'OpDPdyCoarse',
                           'ddy_fine': 'OpDPdyFine', 'DeviceMemoryBarrier': 'OpMemoryBarrier', 
                           'GroupMemoryBarrier': 'OpMemoryBarrier', 'GroupMemoryBarrierWithGroupSync': 'OpControlBarrier',
                           'InterlockedAdd': 'OpAtomicIAdd', 'InterlockedAnd' : 'OpAtomicAnd', 'InterlockedOr' : 'OpAtomicOr',
                           'InterlockedXor' : 'OpAtomicXor', 'InterlockedMin': 'OpAtomicSMin', 'InterlockedMax': 'OpAtomicSMax',
                           'InterlockedCompareStore': 'OpAtomicCompareExchange', 'InterlockedCompareExchange': 'OpAtomicCompareExchange',
                           'InterlockedExchange' : 'OpAtomicExchange', 'dot2add' : 'OpDot', 'dot4add_i8packed': 'OpSDot',
                           'dot4add_u8packed': 'OpUDot','dst' : 'OpFMul', 'isnan': 'OpIsNan', 'isinf': 'OpIsInf',
                           'CallShader': 'OpExecuteCallableKHR', 'ReportHit': 'OpReportIntersectionKHR', 'reversebits': 'OpBitReverse',
                           'WaveIsFirstLane': 'OpGroupNonUniformElect', 'WaveActiveAnyTrue': 'OpGroupNonUniformAny',
                           'WaveActiveAllTrue': 'OpGroupNonUniformAll',  'WaveActiveAllEqual': 'OpGroupNonUniformAllEqual',
                           'WaveActiveBallot': 'OpGroupNonUniformBallot', 'WaveReadLaneAt' : 'OpGroupNonUniformShuffle',
                           'WaveReadLaneFirst': 'OpGroupNonUniformBroadcastFirst', 'WaveActiveCountBits': 'OpGroupNonUniformBallotBitCount',
                           'WaveActiveSum': 'OpGroupNonUniformFAdd', 'WaveActiveProduct': 'OpGroupNonUniformFMul',
                           'WaveActiveBitAnd': 'OpGroupNonUniformBitwiseAnd', 'WaveActiveBitOr': 'OpGroupNonUniformBitwiseOr', 
                           'WaveActiveBitXor': 'OpGroupNonUniformBitwiseXor', 'WaveActiveMin': 'OpGroupNonUniformFMin', 
                           'WaveActiveMax': 'OpGroupNonUniformFMax', 'WavePrefixCountBits': 'OpGroupNonUniformBallotBitCount',
                           'WavePrefixSum': 'OpGroupNonUniformFAdd', 'WavePrefixProduct' : 'OpGroupNonUniformFMul',
                           'WaveMatch': 'OpGroupNonUniformPartitionNV', 'WaveMultiPrefixBitAnd': 'OpGroupNonUniformBitwiseAnd',
                           "WaveMultiPrefixBitOr": 'OpGroupNonUniformBitwiseOr', "WaveMultiPrefixBitXor": 'OpGroupNonUniformBitwiseXor', 
                           "WaveMultiPrefixProduct": 'OpGroupNonUniformFMul', "WaveMultiPrefixSum": 'OpGroupNonUniformFAdd',
                           'QuadReadLaneAt': 'OpGroupNonUniformQuadBroadcast', 'QuadReadAcrossX': 'OpGroupNonUniformQuadSwap',
                           'QuadReadAcrossY': 'OpGroupNonUniformQuadSwap', 'QuadReadAcrossDiagonal': 'OpGroupNonUniformQuadSwap',
                           'WorldToObject4x3' : 'WorldToObjectKHR', 'WorldToObject3x4' : 'WorldToObjectKHR', 'RayFlags' : 'IncomingRayFlagsKHR',
                           'ObjectToWorld4x3' : 'ObjectToWorldKHR', 'ObjectToWorld3x4' : 'ObjectToWorldKHR', 'RayTCurrent' : 'RayTmaxKHR',
                           'DispatchRaysIndex' : 'LaunchIdKHR', 'DispatchRaysDimensions' : 'LaunchSizeKHR', 'InstanceID' : 'InstanceCustomIndexKHR',
                           'GeometryIndex' : 'RayGeometryIndexKHR', 'DispatchMesh' : 'OpEmitMeshTasksEXT', 'SetMeshOutputCounts' : "OpSetMeshOutputsEXT"
                         }

intrinsic_to_dxil_map = {'TraceRay': 'traceRay',
                         'InterlockedExchange': 'atomicBinOp',
                         'InterlockedAnd': 'annotateHandle',
                         'CheckAccessFullyMapped': 'checkAccessFullyMapped',
                         'InterlockedCompareStore': 'atomicCompareExchange'
                         }

# docs say these are numeric, but they are not
float_only_intrinsics = [
    'asfloat16',
    'asint16',
    'asuint16',
    'dst',
    'EvaluateAttributeAtSample',
    'EvaluateAttributeCentroid',
    'EvaluateAttributeSnapped']
DXC_PATH = os.path.join(pathlib.Path().resolve(), 'DXC_Debug_BUILD/bin/dxc')
DXC_COV_PATH = os.path.join(pathlib.Path().resolve(), 'DXC_COV_BUILD/bin/dxc')

isCli_print = False


class TypeIndex(Enum):
    FloatType = 0
    UnsignedType = 1
    SignedType = 2


class VecLength(Enum):
    Vec2 = 2
    Vec3 = 3
    Vec4 = 4


type_qualifer_arr = ['f', 'u', 'i']


def print_cli(*args, **kwargs):
    global isCli_print
    if isCli_print:
        print(' '.join(map(str, args)), **kwargs)


def md_table_to_dict(md_file):
    html_content = pypandoc.convert_file(
        md_file,
        'html',
        format='markdown',
        extra_args=['--standalone'])

    # print(html_content)

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')

    # Extracting headers
    headers = [header.get_text(strip=True) for header in table.find_all('th')]
    name_index = headers.index('Name')
    desc_index = headers.index('Description')
    shader_index = headers.index('Minimum shader model')

    # Extracting data
    data = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        name = cells[name_index].get_text(strip=True)
        description = cells[desc_index].get_text(strip=True)
        # shaders only went from 1-5 before DXC
        # after is 6.0-6.8
        # this modification lets us drop the superscrpt
        shader_version = cells[shader_index].get_text(strip=True)
        if (len(shader_version) > 1 and shader_version[1] != '.'):
            shader_version = cells[shader_index].get_text(strip=True)[0]
        data[name] = [description, shader_version]

    return data

def extract_dxil_opcode(hl_op_name, line):
    match = re.search(r'@dx\.op\..*\(i32\s*(?P<int_opcode>\d+)', line)
    if hl_op_name in intrinsic_to_dxil_map:
        match = re.search(r'@dx\.op\.' +
                          intrinsic_to_dxil_map[hl_op_name] +
                          r'.*\(i32\s*(?P<int_opcode>\d+)', line)
    if match:
        return int(match.group('int_opcode'))
    else:
        return None

def capitalize_and_Op_prefix(word):
    if not word:
        return 'Op'
    capitalized_word = word[0].upper() + word[1:]
    return 'Op' + capitalized_word

def op_prefix_and_ext(word):
    ret_str = capitalize_and_Op_prefix(word)
    return ret_str + 'EXT'

def construct_khr_op(word):
    return word + 'KHR'

def remove_trailing_num(s):
   pattern = r'_\d+$'
   return re.sub(pattern, '', s)

def extract_spirv_opcode(hl_op_name, line):
    match = re.search(r'OpExtInst\s+%\w+\s+%\w+\s+(?P<word_opcode>\w+)\s+%\w+', line)
    if match:
        return match.group('word_opcode')
    
    base_case_hl_op_name = remove_trailing_num(hl_op_name)

    match = re.search(capitalize_and_Op_prefix(base_case_hl_op_name), line)
    if match:
        return match.group(0)
    
    match = re.search(op_prefix_and_ext(base_case_hl_op_name), line)
    if match:
        return match.group(0)
    
    match = re.search(construct_khr_op(base_case_hl_op_name), line)
    if match:
        return match.group(0)

    if hl_op_name in intrinsic_to_spirv_map:
        match = re.search(intrinsic_to_spirv_map[hl_op_name], line)
        if match:
            return match.group(0)
    
    if base_case_hl_op_name in intrinsic_to_spirv_map:
        match = re.search(intrinsic_to_spirv_map[base_case_hl_op_name], line)
        if match:
            return match.group(0)
    else:
        return None

def get_valid_llvm_type(type_name: str, param_name: str = ''):
    if type_name == 'i32' or type_name == 'i8':
        return 'int'
    if type_name == 'f' or type_name == '$o':
        return 'float'
    if type_name == 'i1':
        return 'bool'
    if type_name == 'v':
        return 'void'
    # handle
    if type_name == 'res':
        if param_name == 'accelerationStructure':
            return 'RaytracingAccelerationStructure'
        if param_name == 'rawBuf':
            return 'RWByteAddressBuffer'
    if type_name == '$gsptr':
        return 'groupshared float'
    if (type_name == 'waveMat' and (param_name == 'waveMatrixLeft'
                                    or param_name == 'waveMatrixInput')):
        return 'WaveMatrixLeft<float, 16, 16>'
    if type_name == 'waveMat' and param_name == 'waveMatrixRight':
        return 'WaveMatrixRight<float, 16, 16>'
    if type_name == 'waveMat' and param_name == 'waveMatrixFragment':
        return 'WaveMatrixLeftColAcc<float, 16, 16>'
    if type_name == 'waveMat' and param_name == 'waveMatrixAccumulatorOrFragment':
        return 'WaveMatrixAccumulator<float, 16, 16>'


def get_valid_type(type_name: str, type_index: TypeIndex = TypeIndex.FloatType,
                   vec_length: VecLength = VecLength.Vec4):
    type_name = type_name.strip()
    # case to handle dot2,dot3, dot4
    if type_index == TypeIndex.FloatType and type_name == "numeric<c>":
        return {2: "float2", 3: "float3", 4: "float4"}[vec_length.value]
    if (type_name == "numeric<4>" or type_name == "numeric<c>"
        or type_name == "numeric<>" or type_name == "any<>"
        or type_name == "any_sampler" or type_name == "numeric<c2>"
            or type_name == "numeric<r>"):
        return ["float4", "uint4", "int4"][type_index.value]
    if type_name == "numeric" or type_name == "numeric32_only<>":
        return ["float", "uint", "int"][type_index.value]
    if (type_name == "numeric<r2@c2>" or type_name == "numeric<c2@r2>"
        or type_name == "numeric<r@c>" or type_name == "numeric<c@r>"
        or type_name == "numeric<c@c2>" or type_name == "numeric<r@r2>"
            or type_name == "numeric<r@c2>" or type_name == "numeric<c@r2>"):
        return ["float4x4", "uint4x4", "int4x4"][type_index.value]
    if (type_name == "numeric16_only<>"):
        return ["half4", "uint16_4", "int16_4"][type_index.value]
    if (type_name == "float_like<c>" or type_name == "float_like<4>"
        or type_name == "float<>" or type_name == "float<4>"
            or type_name == "float_like<>" or type_name == "any_float<>"):
        return "float4"
    if type_name == "float<3>" or type_name == "float_like<3>":
        return "float3"
    if type_name == "float<2>":
        return "float2"
    if type_name == "float16_t<>" or type_name == "float16_t<4>":
        return "half4"
    if type_name == "float16_t<3>":
        return "half3"
    if type_name == "float16_t<2>":
        return "half2"
    if (type_name == "double<>" or type_name == "double<4>"
            or type_name == "double_only<>"):
        return "double4"
    if type_name == "double<3>":
        return "double3"
    if type_name == "double<2>":
        return "double2"
    if type_name == "bool<>" or type_name == "bool<4>":
        return "bool4"
    if type_name == "bool<3>":
        return "bool3"
    if type_name == "bool<2>":
        return "bool2"
    if (type_name == "uint<>" or type_name == "uint<4>"
        or type_name == "uint<c>"
            or type_name == "uint_only<>"):
        return "uint4"
    if type_name == "uint<3>":
        return "uint3"
    if type_name == "uint<2>":
        return "uint2"
    if (type_name == "int<>" or type_name == "int<4>"
        or type_name == "any_int<>" or type_name == "sint16or32_only<4>"
            or type_name == "any_int16or32<4>"):
        return "int4"
    if type_name == "p32i8":
        return "int8_t4_packed"
    if type_name == "p32u8":
        return "uint8_t4_packed"
    if type_name == "int<3>":
        return "int3"
    if type_name == "int<2>":
        return "int2"
    if (type_name == "int16_t<4>" or type_name == "int16_t<>"):
        return "int16_t4"
    if type_name == "int16_t<3>":
        return "int16_t3"
    if type_name == "int16_t<2>":
        return "int16_t2"
    if type_name == "uint16_t<4>" or type_name == "uint16_t<>":
        return "uint16_t4"
    if type_name == "uint16_t<3>":
        return "uint16_t3"
    if type_name == "uint16_t<2>":
        return "uint16_t2"
    if (type_name == "float_like"
            or type_name == "float<1>" or type_name == "float32_only"):
        return "float"
    if type_name == "u64":
        return "uint64_t"
    if type_name == "int64_only":
        return "int64_t"
    if type_name == "any_int64":
        return "int64_t"
    if type_name == "uint64_t<4>" or type_name == "uint64_t<>":
        return "uint64_t4"
    if type_name == "uint64_t<3>":
        return "uint64_t3"
    if type_name == "uint64_t<2>":
        return "uint64_t2"
    if type_name == "int64_t<4>" or type_name == "int64_t<>":
        return "int64_t4"
    if type_name == "int64_t<3>":
        return "int64_t3"
    if type_name == "int64_t<2>":
        return "int64_t2"
    if type_name == "float<3@4>":
        return "float3x4"
    if type_name == "float<4@3>":
        return "float4x3"
    if type_name == "acceleration_struct":
        return "RaytracingAccelerationStructure"
    if type_name == "ray_desc":
        return "RayDesc"
    if type_name == "udt":
        return "RayPayload"
    if type_name == "uint_only" or type_name == "NodeRecordOrUAV":
        return "uint"
    if type_name == "int32_only" or type_name == "any_int32":
        return "int"
    if (type_name == "void" or type_name == "int" or type_name == "bool"
            or type_name == "float" or type_name == "uint" or type_name == "string"):
        return type_name


def generate_interlocked(func_name, params, type_index: TypeIndex):
    code_body = f"RWStructuredBuffer<{get_valid_type(params[1].type_name, type_index)}> buffer : register(u0);\n[numthreads(1, 1, 1)]\n"
    return_type = params[0].type_name
    code_body += f"export {get_valid_type(return_type, type_index)} fn(uint3 dispatchThreadID : SV_DispatchThreadID, "
    arg_length = len(params) - 1
    for i in range(1, arg_length):
        code_body += f"{get_valid_type(params[i].type_name, type_index)} p{i}, "
    code_body = code_body.rstrip(", ") + ") {\nint index = dispatchThreadID.x;"

    # Define the function call
    func_call = f"return {func_name}(buffer[index], "
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{code_body}\n    {func_call}"
    return payload


def generate_node(func_name, params, type_index: TypeIndex):
    return_type = params[0].type_name
    arg_length = len(params)
    buffer = f"RWBuffer<{get_valid_type(return_type, type_index)}> buf0;\n"
    fn_header = '[shader("node")]\n[NodeDispatchGrid(1, 1, 1)]\n[numthreads(1, 1, 1)]\nvoid fn() {\n'

    # Define the function call
    func_call = f"buf0[0] = {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{buffer}\n{fn_header}\n{func_call}"
    return payload


def generate_anyhit(func_name, params, type_index: TypeIndex, is_spirv: bool):
    raypayload = 'struct [raypayload] RayPayload\n{\n\tfloat4 color : write(caller) : read(anyhit);\n\tfloat distance : write(caller) : read(anyhit);\n};\n'
    attributes = 'struct Attributes {\n\tfloat3 barycentrics;\n\tuint primitiveIndex;\n};\n'
    shader_header = '[shader("anyhit")]\n'
    shader_header +=  f'{"" if is_spirv else "export "}void fn(inout RayPayload payload, in Attributes attributes) '
    shader_header += '{\n'
    arg_length = len(params)

    # Define the function call
    func_call = f"return {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{raypayload}\n{attributes}\n{shader_header}\t{func_call}"
    return payload


def generate_mesh(func_name, params, type_index: TypeIndex, is_spirv : bool):
    fn_attr = '[numthreads(1, 1, 1)]\n[outputtopology("triangle")]\n[shader("mesh")]'
    fn_sig = 'void fn(in uint gi : SV_GroupIndex, in uint vi : SV_ViewID'
    arg_length = len(params)
    
    if is_spirv:
        fn_datastructure = 'struct MeshPerVertex {\n\tfloat4 position : SV_Position;\n};'
        fn_attr =f'{fn_datastructure}\n{fn_attr}'
        fn_sig += ',\n\tout vertices MeshPerVertex verts[3], out indices uint3 primitiveInd[3]'
    fn_sig += ') {\n'
    func_call = f"{func_name}("
    for i in range(1, arg_length):
        type_name = get_valid_type(params[i].type_name, type_index)
        type_val = int(1)
        if type_name == 'float' or type_name == 'double' or type_name == 'half':
            type_val = float(1.0)
        func_call += f"{type_val}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{fn_attr}\n{fn_sig}\n{func_call}"
    return payload


def generate_amplification(func_name, params, type_index: TypeIndex, is_spirv : bool):
    fn_attr = '[numthreads(1, 1, 1)]\n[shader("amplification")]'
    fn_sig = f'{"" if is_spirv else "export "}'
    fn_sig += 'void fn(uint gtid : SV_GroupIndex) {\n'
    rayPayload = "struct RayPayload\n{\n\tfloat4 color;\n\tfloat distance;\n};"
    fn_data_struct = ""
    rayPayloadInserted = False
    arg_length = len(params)
    arg_list = ""
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name, type_index)
        if not is_spirv and not rayPayloadInserted and arg_type == "RayPayload":
            rayPayloadInserted = True
            arg_list = rayPayload + "\n" + arg_list
        if is_spirv and arg_type == "RayPayload":
            fn_data_struct = f'groupshared {arg_type} p{i};\n'
        else:
            arg_list += f"{arg_type} p{i};\n"
    if fn_data_struct:
        fn_data_struct = f'{rayPayload}\n\n{fn_data_struct}\n'

    # Define the function call
    func_call = f"{func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    # Generate the payload
    payload = f"{fn_data_struct}{fn_attr}\n{fn_sig}\n{arg_list}\n{func_call}"
    return payload

def generate_pixel_spirv(func_name, params, type_index: TypeIndex):
    return generate_pixel(func_name, params, type_index)

def generate_pixel_dxil(func_name, params, type_index: TypeIndex):
    fn_attr = '[numthreads(1, 1, 1)]\n[shader("pixel")]'
    return generate_pixel(func_name, params, type_index,fn_attr)

def generate_pixel(func_name, params, type_index: TypeIndex, fn_attr: str =''):
    global_attr = ""
    
    return_type = params[0].type_name
    signature = f"{get_valid_type(return_type, type_index)} fn("
    if func_name == "GetAttributeAtVertex":
        signature += f"nointerpolation {get_valid_type(params[1].type_name)} p1 : COLOR"
    if func_name in pixel_intrinsics_special:
        signature += f"{get_valid_type(params[1].type_name, type_index)} p1 : COLOR"
    signature += " ) : SV_Target {\n"
    arg_length = len(params)
    if func_name == "GetAttributeAtVertex":
        type_name = params[2].type_name
        type_val = int(1)
        if type_name == 'float' or type_name == 'double' or type_name == 'half':
            type_val = float(1.0)
        arg_type = get_valid_type(type_name, type_index)
        signature += f"\t{arg_type} p{2} = {type_val};"
    else:
        index_start = 1
        if func_name in pixel_intrinsics_special:
            index_start = 2
        for i in range(index_start, arg_length):
            arg_type = get_valid_type(params[i].type_name, type_index)
            signature += f"\t{arg_type} p{i};"

    if func_name == "CheckAccessFullyMapped":
        global_attr = "RWBuffer<float4> g_buffer;\n\n"
        signature += "\n\tfloat4 data = g_buffer.Load(0, p1);"
    # Define the function call
    func_call = ""
    if func_name == "clip":
        func_call = f"\t {func_name}(p1.a"
    else:
        func_call = f"\treturn {func_name}("
        for i in range(1, arg_length):
            func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{global_attr}{fn_attr}\n{signature}\n{func_call}"
    return payload


def generate_hull(func_name, params, type_index: TypeIndex, is_spirv):
    hs_per_patch_data = 'struct HSPerPatchData\n{\n\tfloat edges[ 3 ] : SV_TessFactor;\n\tfloat inside : SV_InsideTessFactor;\n};\n'
    ps_scene_in = 'struct PSSceneIn {\n\tfloat4 pos : SV_Position;\n\tfloat2 tex : TEXCOORD0;\n\tfloat3 norm : NORMAL;\n};\n'
    hs_per_vertex_data = 'struct HSPerVertexData {\n\tPSSceneIn v;\n};\n'
    data_structures = f'{hs_per_patch_data}{ps_scene_in}{hs_per_vertex_data}'
    main_fn_attr = f'[domain("tri")]\n[outputtopology("triangle_cw")]\n[patchconstantfunc("fn")]\n[outputcontrolpoints(3)]\n[partitioning({"\"integer\"" if is_spirv else "\"pow2\""})]'
    main_fn_sig = 'HSPerVertexData main( const uint id : SV_OutputControlPointID,const InputPatch< PSSceneIn, 3 > points ) {'
    main_fn_body = '\n\tHSPerVertexData v;\n\tv.v = points[ id ];\n\treturn v;\n}'

    fn_sig = 'export HSPerPatchData fn( const InputPatch< PSSceneIn, 3 > points ) {\n\tHSPerPatchData d;\n\tfloat4 edgeFactor;\n\tfloat2 insideFactor;'
    fn_body_end = '\n\td.edges[0]=edgeFactor.x;\n\td.edges[1]=edgeFactor.y;\n\td.edges[2]=edgeFactor.z + edgeFactor.w;\n\td.inside = insideFactor.x + insideFactor.y;\n\treturn d;\n}\n'

    arg_length = len(params)
    fn_args = ""
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name, type_index)
        fn_args += f"{arg_type} p{i};\n"

    func_call = f"{func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n"

    fn_test_str = f"{fn_sig}{fn_args}{func_call}{fn_body_end}"
    payload = f"{data_structures}\n{fn_test_str}\n{main_fn_attr}\n{main_fn_sig}\n{main_fn_body}"

    return payload


def generate_raygeneration(func_name, params, type_index: TypeIndex):

    raypayload = 'struct [raypayload] RayPayload\n{\n\tfloat4 color : write(caller) : read(closesthit);\n\tfloat distance : write(caller) : read(closesthit);\n};\n'
    shader_header = '[shader("raygeneration")]\nvoid fn() {'

    arg_length = len(params)
    fn_args = ""
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name, type_index)
        fn_args += f"\t{arg_type} p{i};\n"

    func_call = f"\t{func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    payload = f"{raypayload}\n{shader_header}\n{fn_args}{func_call}"
    return payload

def generate_intersection(func_name, params, type_index: TypeIndex):
    shader_header = '[shader("intersection")]\nvoid fn() {'

    arg_length = len(params)
    fn_args = ""
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name, type_index)
        fn_args += f"\t{arg_type} p{i};\n"

    func_call = f"{func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    return_type = params[0].type_name
    if return_type != 'void':
        func_call = f"{get_valid_type(return_type, type_index)} \tret = " + func_call 
    else:
        func_call = f'\t{func_call}'
    payload = f"{shader_header}\n{fn_args}{func_call}"
    return payload

def generate_scratch_file(
        func_name: str,
        params,
        type_index: TypeIndex = TypeIndex.FloatType,
        vec_length: VecLength = VecLength.Vec4,
        is_spirv: bool = False):
    if func_name.startswith("Interlocked"):
        return generate_interlocked(func_name, params, type_index)
    if func_name in any_hit_intrinsics:
        return generate_anyhit(func_name, params, type_index, is_spirv)
    if func_name in node_intrinsics:
        return generate_node(func_name, params, type_index)
    if func_name in mesh_intrinsics:
        return generate_mesh(func_name, params, type_index, is_spirv)
    if func_name in amplification_intrinsics:
        return generate_amplification(func_name, params, type_index, is_spirv)
    if func_name in pixel_intrinsics:
        return generate_pixel_dxil(func_name, params, type_index)
    if is_spirv and func_name in vulkan_pixel_shader:
        return generate_pixel_spirv(func_name, params, type_index)
    if func_name in hull_intrinsics:
        return generate_hull(func_name, params, type_index, is_spirv)
    if func_name in raygeneration_intrinsics:
        return generate_raygeneration(func_name, params, type_index)
    if is_spirv and func_name in intersection_intrinsics: 
        return generate_intersection(func_name, params, type_index)
    
    rayPayload = "struct RayPayload\n{\n\tfloat4 color;\n\tfloat distance;\n};"
    rayPayloadInserted = False
    return_type = params[0].type_name
    signature = f"export {get_valid_type(return_type, type_index)} fn("
    arg_length = len(params)
    if func_name == "printf":
        arg_length = arg_length - 1
        if is_spirv:
            arg_length = 0
    if func_name in const_intrinsitcs:
        signature = signature + ") {\n"
        for i in range(1, arg_length):
            type_name = params[i].type_name
            type_val = int(1)
            if type_name == 'float' or type_name == 'double' or type_name == 'half':
                type_val = float(1.0)
            arg_type = get_valid_type(type_name, type_index)
            signature += f"\t{arg_type} p{i} = {type_val};\n"
    else:
        for i in range(1, arg_length):
            arg_type = get_valid_type(
                params[i].type_name, type_index, vec_length)
            if not rayPayloadInserted and arg_type == "RayPayload":
                signature = rayPayload + "\n" + signature
            signature += f"{arg_type} p{i}, "
        signature = signature.rstrip(", ") + ") {"

    # Define the function call
    func_call = f"return {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    if is_spirv and func_name == "printf":
        func_call +="\"Hello World\""
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{signature}\n    {func_call}"
    return payload


def printHLSLDoc():
    hlsl_intrinsic_doc_dict = md_table_to_dict(full_md_filepath)
    intrinsics_no_docs = []
    hlsl_intrinsics_count = 0
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in hidden_intrinsics):
            continue
        hlsl_intrinsics_count = hlsl_intrinsics_count + 1
        if hl_op.name in hlsl_intrinsic_doc_dict:
            print(
                f'{hl_op.name} args {len(hl_op.params)} docs: {hlsl_intrinsic_doc_dict[hl_op.name]}')
        else:
            intrinsics_no_docs.append(hl_op.name)
    print("no docs:")
    print(intrinsics_no_docs)
    print(
        f'percentage without docs: {100.0 * (len(intrinsics_no_docs)/ hlsl_intrinsics_count)}%')


def build_dxc(rebuild: bool = False, cov_build : bool = False):
    if not rebuild and ((cov_build and os.path.exists(DXC_COV_PATH)) or 
                        (not cov_build and os.path.exists(DXC_PATH))):
        print("Build Succeeded!")
        return
    
    build_dir_name ='DXC_Debug_BUILD'
    if cov_build:
        build_dir_name = 'DXC_COV_BUILD' 

    if rebuild:
        try:
            shutil.rmtree(
                os.path.join(
                    pathlib.Path().resolve(),
                    build_dir_name))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    cmake_dxc_build_command = [
        "cmake",
        "-S DirectXShaderCompiler",
        f"-B {build_dir_name}",
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DLLVM_ENABLE_LLD=ON",
        "-G Ninja",
        "-C DirectXShaderCompiler/cmake/caches/PredefinedParams.cmake"
    ]

    ninja_dxc_build_command = ["ninja", "-C", build_dir_name]

    if cov_build:
        cmake_dxc_build_command.append('-DLLVM_BUILD_INSTRUMENTED_COVERAGE=TRUE')

    result = subprocess.run(
        cmake_dxc_build_command,
        capture_output=True,
        text=True)
    if result.returncode == 0:
        print("CMake Succeeded!")
    else:
        print("CMake Failed!")
        print(result.stderr)
        # printverbose(result.stdout)
        return

    result = subprocess.run(
        ninja_dxc_build_command,
        capture_output=True,
        text=True)
    if result.returncode == 0:
        print("Build Succeeded!")
    else:
        print("Build Failed!")
        print(result.stderr)
        # printverbose(result.stdout)


def get_unique_name(op_name, name_count) -> str:
    if op_name in name_count:
        name_count[op_name] = name_count[op_name] + 1
    else:
        name_count[op_name] = 0
    return f"{op_name}{'' if name_count[op_name] == 0 else f'_{name_count[op_name]}'}"


def check_for_numeric_params(hl_op):
    numeric_found = False
    for param in hl_op.params:
        type_name = param.type_name
        if (type_name.startswith("numeric")):
            numeric_found = True
            break
    return numeric_found


def run_hlsl_test(dxc_command, hlsl_name, hlsl_construct_to_opcode, fail_list, is_spirv=False):
    result = subprocess.run(dxc_command, capture_output=True, text=True)
    if result.returncode == 0:
        opcode = None
        if is_spirv:
            opcode = extract_spirv_opcode(hlsl_name, result.stdout)
        else:
            opcode = extract_dxil_opcode(hlsl_name, result.stdout)
        #if hlsl_name == 'QuadReadAcrossDiagonal':
        #    print(opcode)
        #    print(result.stdout)
        #    exit(0)
        if (opcode is None):
            hlsl_construct_to_opcode[hlsl_name] = -1
        else:
            if not is_spirv:
                opcode = int(opcode)
            hlsl_construct_to_opcode[hlsl_name] = opcode
    else:
        print_cli(result.stderr)
        fail_list.append(hlsl_name)


def dxc_intrinsic_run_helper(
        hl_op,
        dxc_command,
        scratchpad_path,
        hl_op_name,
        intrinsic_to_opcode,
        fail_list,
        type_index: TypeIndex = TypeIndex.FloatType,
        vec_length: VecLength = VecLength.Vec4):
    scratchpad = os.path.join(scratchpad_path, hl_op_name + "_test.hlsl")
    dxc_command[1] = scratchpad
    if hl_op.name in hull_intrinsics:
        dxc_command[2] = "-T hs_6_8"
    else:
        dxc_command[2] = "-T lib_6_8"
    payload = generate_scratch_file(
        hl_op.name, hl_op.params, type_index, vec_length)
    write_payload(scratchpad, payload)
    run_hlsl_test(dxc_command, hl_op_name, intrinsic_to_opcode, fail_list)


def dxc_intrinsic_spirv_run_helper(
        hl_op,
        dxc_command,
        scratchpad_path,
        hl_op_name,
        intrinsic_to_opcode,
        fail_list,
        type_index: TypeIndex = TypeIndex.FloatType,
        vec_length: VecLength = VecLength.Vec4):
    scratchpad = os.path.join(scratchpad_path, hl_op_name + "_spirv_test.hlsl")
    dxc_command[1] = scratchpad
    if hl_op.name in hull_intrinsics:
        dxc_command[2] = "-T hs_6_8"
        dxc_command[3] = "-E main"
    elif hl_op.name in vulkan_pixel_shader:
        dxc_command[2] = "-T ps_6_8"
        dxc_command[3] = "-E fn"
    else:
        dxc_command[2] = "-T lib_6_8"
    payload = generate_scratch_file(
        hl_op.name, hl_op.params, type_index, vec_length, is_spirv=True)
    write_payload(scratchpad, payload)
    run_hlsl_test(dxc_command, hl_op_name, intrinsic_to_opcode, fail_list, is_spirv=True)


# Run dxc
def run_dxc():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    deprecated_intrinsics = gen_deprecated_intrinsics()
    intrinsic_to_opcode = {}
    fail_list = []
    name_count = {}
    total_intrinsics = 0
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
                hl_op.name in hidden_intrinsics):
            continue
        total_intrinsics = total_intrinsics + 1
        hl_op_name = get_unique_name(hl_op.name, name_count)
        numeric_found = check_for_numeric_params(hl_op)
        if hl_op.name == "WaveMatch":
            dxc_command[4] = "-O1"
        else:
            dxc_command[4] = "-O0"
        if numeric_found and hl_op.name == "dot":
            for vec_len in VecLength:
                dxc_intrinsic_run_helper(
                    hl_op,
                    dxc_command,
                    scratchpad_path,
                    hl_op_name,
                    intrinsic_to_opcode,
                    fail_list,
                    TypeIndex.FloatType,
                    vec_len)
                hl_op_name = get_unique_name(hl_op.name, name_count)
        if numeric_found and hl_op.name not in float_only_intrinsics:
            for type_index in TypeIndex:
                if hl_op.name == "dot" and type_index == TypeIndex.FloatType:
                    continue
                dxc_intrinsic_run_helper(
                    hl_op,
                    dxc_command,
                    scratchpad_path,
                    hl_op_name,
                    intrinsic_to_opcode,
                    fail_list,
                    type_index)
                hl_op_name = get_unique_name(hl_op.name, name_count)
        else:
            dxc_intrinsic_run_helper(
                hl_op,
                dxc_command,
                scratchpad_path,
                hl_op_name,
                intrinsic_to_opcode,
                fail_list)

    print_cli(
        f"success %: {100.0 * ( (total_intrinsics - len(fail_list)) / total_intrinsics)}")
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    return intrinsic_to_opcode


def formatShaderModel(smodel):
    return "{}.{}".format(smodel[0], smodel[1])


def query_dxil():
    opcode_to_dxil_docs = {}
    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op:
            opcode_to_dxil_docs[dxil_inst.dxil_opid] = [dxil_inst.dxil_op, formatShaderModel(
                dxil_inst.shader_model), dxil_inst.category, dxil_inst.shader_stages, dxil_inst.doc]
            print_cli(
                dxil_inst.dxil_op,
                dxil_inst.dxil_opid,
                formatShaderModel(
                    dxil_inst.shader_model),
                dxil_inst.shader_stages,
                dxil_inst.doc)
    return opcode_to_dxil_docs


def get_all_types():
    type_set = set()
    for hl_op in db_hlsl.intrinsics:
        for param in hl_op.params:
            type_set.add(param.type_name)
    print(type_set)


def get_all_intrinsic_types():
    type_set = set()
    for hl_op in db_hlsl.intrinsics:
        if hl_op.ns != "Intrinsics":
            continue
        print(f"{hl_op.name}:")
        for param in hl_op.params:
            print(f"\t{param.name} : {param.type_name}")


def get_specifc_intrinsic_types(func_name: str):
    type_set = set()
    for hl_op in db_hlsl.intrinsics:
        if (func_name != hl_op.name):
            continue
        print(f"{hl_op.name}:")
        for param in hl_op.params:
            print(f"\t{param.name} : {param.type_name}")


def dxc_shader_model_helper(dxc_sm):
    if dxc_sm == "":
        return "6.0"
    return dxc_sm


def hlsl_shader_model_helper(dxc_sm, doc_sm):
    if doc_sm == "":
        return dxc_shader_model_helper(dxc_sm)
    return doc_sm


def hlsl_docs_helper(hlsl_intrinsic_doc_dict, intrinsic_name):
    intrinsic_name = intrinsic_name.split('_')[0]
    return hlsl_intrinsic_doc_dict[intrinsic_name] if intrinsic_name in hlsl_intrinsic_doc_dict else [
        "", ""]


def gen_csv_data_base(
        hlsl_to_dxil_op,
        dxil_op_to_docs,
        hlsl_intrinsic_doc_dict):
    data = [['HLSL Intrinsic',
             'DXIL Opcode',
             'DXIL OpName',
             "Shader Model",
             "Shader Model DXC",
             "Shader Stages",
             "Shader Category",
             "DXIL Docs",
             "HLSL Docs"]]

    for key, value in hlsl_to_dxil_op.items():
        dxil_docs = dxil_op_to_docs[value] if value != - \
            1 else ["", "", "", "", ""]
        hlsl_docs = hlsl_docs_helper(hlsl_intrinsic_doc_dict, key)
        row = [
            key,
            value,
            dxil_docs[0],
            hlsl_shader_model_helper(
                dxil_docs[1],
                hlsl_docs[1]),
            dxc_shader_model_helper(
                dxil_docs[1]),
            dxil_docs[2],
            dxil_docs[3],
            dxil_docs[4],
            hlsl_docs[0]]
        data.append(row)
    return data


def gen_csv_data():
    hlsl_to_dxil_op = run_dxc()
    dxil_op_to_docs = query_dxil()
    hlsl_intrinsic_doc_dict = md_table_to_dict(full_md_filepath)
    return gen_csv_data_base(
        hlsl_to_dxil_op,
        dxil_op_to_docs,
        hlsl_intrinsic_doc_dict)


def gen_csv(csv_file_path: str):
    data = gen_csv_data()
    # Open the file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def get_missing_opcodes_from_docs(hlsl_to_dxil_op, dxil_op_to_docs):
    copied_dxil_op_to_docs = dxil_op_to_docs.copy()
    for key, value in hlsl_to_dxil_op.items():
        if value in copied_dxil_op_to_docs:
            del copied_dxil_op_to_docs[value]
    return copied_dxil_op_to_docs


def gen_dxil_ops_as_table(opcodes):
    data = [['DXIL Opcode', 'DXIL OpName', "Shader Model DXC",
             "Shader Category", "Shader Stages", "DXIL Docs"]]
    for key, value in opcodes.items():
        row = [
            key,
            value[0],
            dxc_shader_model_helper(
                value[1]),
            value[2],
            value[3],
            value[4]]
        data.append(row)
    return data


def gen_remaining_opcode_data(
        hlsl_to_dxil_op,
        dxil_op_to_docs,
        semantic_to_dxil_op=None,
        rayquery_to_dxil_op=None,
        wavemat_to_dxil_op=None,
        resources_to_dxil_op=None,
        resources_sample_to_dxil_op=None,
        sampler_feedback_to_opcode=None,
        mesh_shader_instr_to_opcode=None,
        geo_instr_to_opcode=None,
        hull_instr_to_opcode=None,
        node_instr_to_opcode=None,
        texture_gather_instr_to_opcode=None):
    opcode_count = len(dxil_op_to_docs)
    rem_opcodes = get_missing_opcodes_from_docs(
        hlsl_to_dxil_op, dxil_op_to_docs)
    if semantic_to_dxil_op:
        rem_opcodes = get_missing_opcodes_from_docs(
            semantic_to_dxil_op, rem_opcodes)
    if rayquery_to_dxil_op:
        rem_opcodes = get_missing_opcodes_from_docs(
            rayquery_to_dxil_op, rem_opcodes)
    if wavemat_to_dxil_op:
        rem_opcodes = get_missing_opcodes_from_docs(
            wavemat_to_dxil_op, rem_opcodes)
    if resources_to_dxil_op:
        rem_opcodes = get_missing_opcodes_from_docs(
            resources_to_dxil_op, rem_opcodes)
    if resources_sample_to_dxil_op:
        rem_opcodes = get_missing_opcodes_from_docs(
            resources_sample_to_dxil_op, rem_opcodes)
    if sampler_feedback_to_opcode:
        rem_opcodes = get_missing_opcodes_from_docs(
            sampler_feedback_to_opcode, rem_opcodes)
    if mesh_shader_instr_to_opcode:
        rem_opcodes = get_missing_opcodes_from_docs(
            mesh_shader_instr_to_opcode, rem_opcodes)
    if geo_instr_to_opcode:
        rem_opcodes = get_missing_opcodes_from_docs(
            geo_instr_to_opcode, rem_opcodes)
    if hull_instr_to_opcode:
        rem_opcodes = get_missing_opcodes_from_docs(
            hull_instr_to_opcode, rem_opcodes)
    if node_instr_to_opcode:
        rem_opcodes = get_missing_opcodes_from_docs(
            node_instr_to_opcode, rem_opcodes)
    if texture_gather_instr_to_opcode:
        rem_opcodes = get_missing_opcodes_from_docs(
            texture_gather_instr_to_opcode, rem_opcodes)

    return gen_dxil_ops_as_table(rem_opcodes)


def get_found_ops(dxil_ops, dxil_op_to_docs):
    found_opcodes = {}
    for key, value in dxil_ops.items():
        if value in dxil_op_to_docs:
            found_opcodes[value] = dxil_op_to_docs[value]
    return found_opcodes


def gen_found_opcodes(dxil_ops, dxil_op_to_docs):
    found_opcodes = get_found_ops(dxil_ops, dxil_op_to_docs)
    return gen_dxil_ops_as_table(found_opcodes)


def gen_arg_semantics(argSemantic: str):
    fn_datastruct = 'RWStructuredBuffer<float> buffer : register(u0);\n'
    fn_attr = '[numthreads(1, 1, 1)]\n[outputtopology("triangle")]\n[shader("mesh")]'
    fn_sig = f'void fn(uint i : {argSemantic}'
    fn_body = '{\n\tbuffer[i] = 1;'
    semantic_to_type = {
        "SV_ClipDistance": "float2",
        "SV_CullDistance": "float2",
        "SV_Barycentrics": "float3",
        "SV_StencilRef": "out uint",
        "SV_DepthLessEqual": "out float",
        "SV_Depth": "out float"}

    if argSemantic in [
        "SV_ViewID",
        "SV_Coverage",
        "SV_InnerCoverage",
        "SV_RenderTargetArrayIndex",
        "SV_CullPrimitive",
        "SV_ShadingRate",
        "SV_StencilRef",
        "SV_InstanceID",
        "SV_DepthLessEqual",
        "SV_Depth",
        "SV_IsFrontFace",
        "SV_SampleIndex",
        "SV_ClipDistance",
        "SV_CullDistance",
            "SV_Barycentrics"]:
        fn_attr = '[shader("pixel")]'
    if argSemantic in [
        "SV_ClipDistance",
        "SV_CullDistance",
            "SV_Barycentrics"]:
        fn_sig = f'void fn({semantic_to_type[argSemantic]} uv : {argSemantic}'
        fn_body = '{\n\tbuffer[0] = uv.x;'
    if argSemantic in ["SV_StencilRef", "SV_DepthLessEqual", "SV_Depth"]:
        fn_sig = f'void fn({semantic_to_type[argSemantic]} i : {argSemantic}'
        fn_body = '{\n\t i = buffer[0];'
    if argSemantic in ["SV_IsFrontFace"]:
        fn_body = '{\n\tbuffer[0] = i;'
    if argSemantic in [
        "SV_StartVertexLocation",
        "SV_StartInstanceLocation",
        "SV_ViewPortArrayIndex",
            "SV_VertexID"]:
        fn_attr = '[shader("vertex")]'
    if argSemantic in ["SV_DomainLocation"]:
        fn_attr = '[shader("domain")]\n[domain("quad")]'
        fn_sig = f'void fn(float2 uv : {argSemantic}'
        fn_body = '{\n\tbuffer[0] = uv.x;'
    if argSemantic in ["SV_TessFactor", "SV_InsideTessFactor"]:
        fn_attr = '[shader("domain")]\n[domain("isoline")]'
        fn_sig = f'void fn(float uv[2] : {argSemantic}'
        fn_body = '{\n\tbuffer[0] = uv[0];'
    if argSemantic in ["SV_InsideTessFactor"]:
        fn_attr = '[shader("domain")]\n[domain("tri")]'
        fn_sig = f'void fn(float uv : {argSemantic}'
        fn_body = '{\n\tbuffer[0] = uv;'
    if argSemantic in ["SV_GSInstanceID", "SV_PrimitiveID"]:
        fn_datastruct += 'struct PosStruct {\n\tfloat4 pos : SV_Position;\n};\n'
        fn_attr = '[shader("geometry")]\n[maxvertexcount(1)]'
        fn_sig += ', inout PointStream<PosStruct> OutputStream'
        fn_body += '\n\tPosStruct s;\n\ts.pos = 1;\n\tOutputStream.Append(s);'

    if argSemantic in ["SV_OutputControlPointID"]:
        fn_datastruct += 'struct HSPerPatchData {\n\tfloat edges[3] : SV_TessFactor;\n\tfloat inside : SV_InsideTessFactor;\n};\n'
        fn_datastruct += '\nHSPerPatchData HSPerPatchFunc() {\n\tHSPerPatchData d;\n\treturn d;\n}\n'
        fn_attr = '[shader("hull")]\n[domain("tri")]\n[partitioning("fractional_odd")]\n[outputtopology("triangle_cw")]\n[patchconstantfunc("HSPerPatchFunc")]\n[outputcontrolpoints(3)]'
    fn_sig += ')'
    fn_body += '\n}'
    # Generate the payload
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    return payload


def gen_trace_rayline(params):
    ray_struct = ""
    for i in range(6, len(params)):
        arg_type = get_valid_llvm_type(params[i].llvm_type, params[i].name)
        ray_struct += f"\t{arg_type} {params[i].name};\n"
    ray_struct += f'\tRayDesc ray;\n\tray.TMin = {params[9].name};\n\tray.TMax = {params[13].name};\n'
    ray_struct += f'\tray.Origin = float3({params[6].name},{params[7].name},{params[8].name});\n'
    ray_struct += f'\tray.Direction = float3({params[10].name},{params[11].name},{params[12].name});\n'
    return (6, ray_struct)


def gen_wavematrix():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    wave_mat_main_types = ["Right", "Left"]
    wave_mat_acc_types = ["Accumulator"]
    wave_mat_linear_acc_types = ["LeftColAcc", "RightRowAcc"]
    wave_mat_types = wave_mat_main_types
    scalar_funcs = [
        "ScalarMultiply",
        "ScalarDivide",
        "ScalarAdd",
        "ScalarSubtract"]
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0",
        "-Vd"  # There is not shader model 6.9 yet so need to turn validator off
    ]
    unique_names = {}
    wave_query_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        global_vars = ""
        start_index = 3
        if dxil_inst.dxil_op and dxil_inst.category == "WaveMatrix":
            if dxil_inst.name == "WaveMatrix_Annotate":
                continue
            func_name = ("Matrix" if dxil_inst.name == "WaveMatrix_Depth" else "") + \
                dxil_inst.name.split("WaveMatrix_")[-1]
            if func_name in ["LoadRawBuf", "LoadGroupShared"]:
                func_name = "Load"
            elif func_name in ["StoreRawBuf", "StoreGroupShared"]:
                func_name = "Store"
            elif func_name == "ScalarOp":
                func_name = scalar_funcs[0]
                start_index = 4
                wave_mat_types = wave_mat_acc_types + wave_mat_linear_acc_types
            elif func_name == "MultiplyAccumulate":
                wave_mat_types = wave_mat_acc_types
            elif func_name.startswith("Multiply") or func_name == "Add":
                wave_mat_types = wave_mat_acc_types
            elif func_name == "SumAccumulate":
                wave_mat_types = wave_mat_linear_acc_types
            else:
                wave_mat_types = wave_mat_main_types
            params = dxil_inst.ops
            # subtract one for return and one for dxil opcode
            arg_length = len(params)
            fn_args = ""
            ret_val = ""
            shader_header = '\n[shader("compute")]\n[numthreads(1, 1, 1)]\n void fn() {'
            if params[0].llvm_type != "v":
                global_vars = "RWByteAddressBuffer rwbuf;\n"
                func_call = f"\t{get_valid_llvm_type(params[0].llvm_type, params[0].name)} retVal = waveMat.{func_name}("
                ret_val = "\n\trwbuf.Store(0, retVal);"
            else:
                func_call = f"\twaveMat.{func_name}("
            for i in range(start_index, arg_length):
                arg_type = get_valid_llvm_type(
                    params[i].llvm_type, params[i].name)
                if params[i].llvm_type == "$gsptr":
                    global_vars += f"\n{arg_type} {params[i].name}[5];\n"
                else:
                    fn_args += f"\n\t{arg_type} {params[i].name};\n"
                func_call += f"{params[i].name}, "
            func_call = func_call.rstrip(", ") + ");\n"
            for wave_mat_type in wave_mat_types:
                fn_wave_mat = f'\tWaveMatrix{wave_mat_type}<float, 16, 16> waveMat;\n'
                if wave_mat_type == "RightRowAcc":
                    fn_args = fn_args.replace(
                        "WaveMatrixLeft", "WaveMatrixRight")
                payload = f"{global_vars}\n{shader_header}{fn_args}\n{fn_wave_mat}{func_call}{ret_val}\n"
                payload += "}"
                unique_name = get_unique_name(dxil_inst.name, unique_names)
                scratchpad = os.path.join(
                    scratchpad_path, f"{unique_name}_test.hlsl")
                dxc_command[1] = scratchpad
                write_payload(scratchpad, payload)
                if dxil_inst.name not in intrinsic_to_dxil_map:
                    if (dxil_inst.name == "WaveMatrix_Depth"):
                        intrinsic_to_dxil_map[dxil_inst.name] = "waveMatrix_Depth"
                    elif func_name in scalar_funcs:
                        intrinsic_to_dxil_map[dxil_inst.name] = "waveMatrix_ScalarOp"
                    elif func_name in ["Add", "SumAccumulate"]:
                        intrinsic_to_dxil_map[dxil_inst.name] = "waveMatrix_Accumulate"
                    elif func_name == "MultiplyAccumulate":
                        intrinsic_to_dxil_map[dxil_inst.name] = "waveMatrix_Multiply"
                    else:
                        intrinsic_to_dxil_map[dxil_inst.name] = "waveMatrix_" + func_name
                run_hlsl_test(
                    dxc_command,
                    unique_name,
                    wave_query_to_opcode,
                    fail_list)

    if (len(fail_list) > 0):
        lf = len(fail_list)
        ls = len(wave_query_to_opcode)
        print_cli(f"FAILED Tests: {lf}")
        print_cli(fail_list)
        print_cli(f"FAILED Percentage: {100* (lf / (lf+ls))}%")
    else:
        print_cli(f"100% of tests passed!")
        print_cli(wave_query_to_opcode)
    return wave_query_to_opcode


def gen_ray_query():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    ray_query_flags = [
        'RAY_FLAG_CULL_NON_OPAQUE',
        'RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES',
        'RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH']
    # llvm_types = set()
    unique_names = {}
    ray_query_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        global_vars = ""
        if dxil_inst.dxil_op and dxil_inst.category == "Inline Ray Query":
            if dxil_inst.name == "AllocateRayQuery":
                continue
            func_name = dxil_inst.name.split("RayQuery_")[-1]
            params = dxil_inst.ops
            # subtract one for return and one for dxil opcode
            arg_length = len(params)
            fn_args = ""
            shader_header = f'\n[shader("pixel")]\n{get_valid_llvm_type(params[0].llvm_type)} fn() {"" if params[0].llvm_type == "v" else ": SV_Target"}'
            # llvm_types.add(params[0].llvm_type)
            shader_header = shader_header + " {\n"
            func_call = f"\treturn query.{func_name}("
            if ("TraceRayInline" == func_name):
                (arg_length, fn_args) = gen_trace_rayline(params)
            if (func_name.startswith("Candidate") or func_name.startswith(
                    "Committed") or func_name.startswith("WorldRay")):
                arg_length = 2
            for i in range(2, arg_length):
                if params[i].name == "rayQueryHandle":
                    continue
                # llvm_types.add(params[i].llvm_type)
                arg_type = get_valid_llvm_type(
                    params[i].llvm_type, params[i].name)
                if arg_type in ["RaytracingAccelerationStructure"]:
                    global_vars += f"{arg_type} {params[i].name} : register(t0, space0);\n"
                else:
                    fn_args += f"\t{arg_type} {params[i].name};\n"
                func_call += f"{params[i].name}, "
            if ("TraceRayInline" == func_name):
                func_call += f"ray, "
            func_call = func_call.rstrip(", ") + ");\n}"
            for ray_query_flag in ray_query_flags:
                fn_rayQuery = f'\tRayQuery<{ray_query_flag}> query;\n'
                payload = f"{global_vars}\n{shader_header}{fn_args}\n{fn_rayQuery}{func_call}"
                unique_name = get_unique_name(dxil_inst.name, unique_names)
                scratchpad = os.path.join(
                    scratchpad_path, f"{unique_name}_test.hlsl")
                dxc_command[1] = scratchpad
                write_payload(scratchpad, payload)
                if dxil_inst.name not in intrinsic_to_dxil_map:
                    if (func_name.startswith("World") or func_name.endswith("RayOrigin") or
                            func_name.endswith("RayDirection") or func_name.endswith("Barycentrics")):
                        intrinsic_to_dxil_map[dxil_inst.name] = "rayQuery_StateVector"
                    elif (func_name.startswith("CandidateObject") or func_name.startswith("CandidateWorld") or
                          func_name.startswith("CommittedObject") or func_name.startswith("CommittedWorld")):
                        intrinsic_to_dxil_map[dxil_inst.name] = "rayQuery_StateMatrix"
                    elif (func_name.startswith("Candidate") or func_name.startswith("Committed") or
                          func_name.startswith("RayFlags") or func_name.startswith("RayTMin")):
                        intrinsic_to_dxil_map[dxil_inst.name] = "rayQuery_StateScalar"
                    else:
                        intrinsic_to_dxil_map[dxil_inst.name] = "rayQuery_" + func_name
                run_hlsl_test(
                    dxc_command,
                    unique_name,
                    ray_query_to_opcode,
                    fail_list)

    if (len(fail_list) > 0):
        lf = len(fail_list)
        ls = len(ray_query_to_opcode)
        print_cli(f"FAILED Tests: {lf}")
        print_cli(fail_list)
        print_cli(f"FAILED Percentage: {100* (lf / (lf+ls))}%")
    else:
        print_cli(f"100% of tests passed!")
        print_cli(ray_query_to_opcode)
    return ray_query_to_opcode


def shader_semantic_table_to_dict_helper(table, col_name: str):
    # Extracting headers
    headers = [header.get_text(strip=True) for header in table.find_all('th')]
    name_index = headers.index(col_name)
    desc_index = headers.index('Description')
    type_index = headers.index('Type')

    # Extracting data
    data = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        name = cells[name_index].get_text(strip=True)
        description = cells[desc_index].get_text(strip=True)
        semantic_type = cells[type_index].get_text(strip=True)
        data[name] = [description, semantic_type]

    return data


def shader_semantic_md_table_to_dict(md_file):
    html_content = pypandoc.convert_file(
        md_file,
        "html",
        format="markdown",
        extra_args=["--standalone"])

    # print(html_content)

    soup = BeautifulSoup(html_content, 'html.parser')
    vertex_shader_h3_tag = soup.find('h3', id='vertex-shader-semantics')
    vertex_semantic_input_table = vertex_shader_h3_tag.find_next_sibling(
        'table')
    vertex_semantic_output_table = vertex_semantic_input_table.find_next_sibling(
        'table')

    pixel_shader_h3_tag = soup.find('h3', id='pixel-shader-semantics')
    pixel_semantic_input_table = pixel_shader_h3_tag.find_next_sibling('table')
    pixel_semantic_output_table = pixel_semantic_input_table.find_next_sibling(
        'table')

    return {
        "vertex":
        [
            shader_semantic_table_to_dict_helper(vertex_semantic_input_table, "Input"),
            shader_semantic_table_to_dict_helper(vertex_semantic_output_table, "Output")
        ],
        "pixel":
        [
            shader_semantic_table_to_dict_helper(pixel_semantic_input_table, "Input"),
            shader_semantic_table_to_dict_helper(pixel_semantic_output_table, "Output")
        ]
    }


def gen_input_semantic_test_case(shader_model: str, type_str, semantic: str):
    out_put_semantic = 'SV_ClipDistance' if shader_model == 'vertex' else 'SV_Target'
    retType = type_str
    if semantic == 'BLENDINDICES':
        retType = 'float'
    shader = f'[shader("{shader_model}")]\n {retType} fn({type_str} p0 : {semantic}) : {out_put_semantic}'
    shader = shader + ' { return p0; }'
    return shader


def gen_output_semantic_test_case(shader_model: str, type_str, semantic: str):
    shader = f'[shader("{shader_model}")]\n {type_str} fn({type_str} p0 : COLOR) : {semantic}'
    shader = shader + ' { return p0; }'
    return shader


def write_payload(scratchpad, payload):
    with open(scratchpad, "w") as file:
        file.write(payload)


def gen_semantics_hlsl():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    semantics = db_dxil.enums[0]
    for enum in db_dxil.enums:
        if "SemanticKind" == enum.name:
            semantics = enum
    semanticsArgsMap = {
        "Barycentrics": "SV_Barycentrics",
        "ClipDistance": "SV_ClipDistance",
        "Coverage": "SV_Coverage",
        "CullDistance": "SV_CullDistance",
        "CullPrimitive": "SV_CullPrimitive",
        "Depth": "SV_Depth",
        "DepthLessEqual": "SV_DepthLessEqual",
        "DispatchThreadID": "SV_DispatchThreadID",
        "DomainLocation": "SV_DomainLocation",
        "GroupID": "SV_GroupID",
        "GroupIndex": "SV_GroupIndex",
        "GroupThreadID": "SV_GroupThreadID",
        "GSInstanceID": "SV_GSInstanceID",
        "InnerCoverage": "SV_InnerCoverage",
        "InstanceID": "SV_InstanceID",
        "InsideTessFactor": "SV_InsideTessFactor",
        "IsFrontFace": "SV_IsFrontFace",
        "OutputControlPointID": "SV_OutputControlPointID",
        "PrimitiveID": "SV_PrimitiveID",
        "RenderTargetArrayIndex": "SV_RenderTargetArrayIndex",
        "SampleIndex": "SV_SampleIndex",
        "ShadingRate": "SV_ShadingRate",
        "StartVertexLocation": "SV_StartVertexLocation",
        "StartInstanceLocation": "SV_StartInstanceLocation",
        "StencilRef": "SV_StencilRef",
        "TessFactor": "SV_TessFactor",
        "VertexID": "SV_VertexID",
        "ViewID": "SV_ViewID",
        "ViewPortArrayIndex": "SV_ViewPortArrayIndex"
    }
    semantic_to_opcode = {}
    fail_list = []

    # ShaderSemantics from docs
    shader_semantics = shader_semantic_md_table_to_dict(
        shader_semantic_filepath)
    for shader_model, semantic_value in shader_semantics.items():
        input_semantics = semantic_value[0]
        output_semantics = semantic_value[1]
        for semantic_name, semantic_attributes in input_semantics.items():
            s_name = semantic_name.split('[')[0]
            s_type = semantic_attributes[1]
            test_str = gen_input_semantic_test_case(
                shader_model, s_type, s_name)
            unique_name = f"{s_name}_{shader_model}_input"
            scratchpad = os.path.join(
                scratchpad_path, f"{unique_name}_test.hlsl")
            dxc_command[1] = scratchpad
            write_payload(scratchpad, test_str)
            intrinsic_to_dxil_map[unique_name] = 'loadInput'
            run_hlsl_test(
                dxc_command,
                unique_name,
                semantic_to_opcode,
                fail_list)

        for semantic_name, semantic_attributes in output_semantics.items():
            s_name = semantic_name.split('[')[0]
            s_type = semantic_attributes[1]
            test_str = gen_input_semantic_test_case(
                shader_model, s_type, s_name)
            unique_name = f"{s_name}_{shader_model}_output"
            scratchpad = os.path.join(
                scratchpad_path, f"{unique_name}_test.hlsl")
            dxc_command[1] = scratchpad
            write_payload(scratchpad, test_str)
            intrinsic_to_dxil_map[unique_name] = 'storeOutput'
            run_hlsl_test(
                dxc_command,
                unique_name,
                semantic_to_opcode,
                fail_list)

    for semantic in semantics.values:
        if semantic.name in semanticsArgsMap:
            scratchpad = os.path.join(
                scratchpad_path, semantic.name + "_test.hlsl")
            dxc_command[1] = scratchpad
            hlsl_name = semanticsArgsMap[semantic.name]
            payload = gen_arg_semantics(hlsl_name)
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                hlsl_name,
                semantic_to_opcode,
                fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(semantic_to_opcode)
    return semantic_to_opcode


def gen_resources_test(dxil_inst_name, fn_datastruct, dxil_op_name,
                       fn_body=None):

    fn_attr = '[shader("pixel")]'
    fn_sig = 'uint fn() : SV_Target {'
    if fn_body is None:
        fn_body = '\n\treturn buffer[0];\n}'
    if dxil_op_name:
        intrinsic_to_dxil_map[dxil_inst_name] = dxil_op_name
    if dxil_inst_name.endswith("Store"):
        fn_sig = f'void fn(uint{"2" if dxil_op_name == "textureStore" else ""} i : SV_InstanceID) '
        fn_sig += "{"
        fn_body = '\n\tbuffer[i] = 1;\n}'
    if dxil_inst_name == "TextureStoreSample":
        fn_sig = 'float4 fn(uint sampleSlice : S, uint2 pos2 : PP) : SV_Target {'
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    return payload


def gen_raw_buffer_resources(dxil_inst_name):
    # default behavior is load
    fn_datastruct = 'RWStructuredBuffer<uint> buffer : register(u0);\n'
    dxil_op_name = 'rawBufferLoad'
    if dxil_inst_name.endswith("Store"):
        dxil_op_name = 'rawBufferStore'
    return gen_resources_test(dxil_inst_name, fn_datastruct, dxil_op_name)


def gen_reg_buffer_resources(dxil_inst_name):
    fn_datastruct = 'Buffer<uint> buffer;\n'
    dxil_op_name = 'bufferLoad'
    # note:  Buffer is a read only store does not make sense
    if dxil_inst_name.endswith("Store"):
        fn_datastruct = 'RWBuffer<uint> buffer;'
        dxil_op_name = 'bufferStore'
    return gen_resources_test(dxil_inst_name, fn_datastruct, dxil_op_name)


def gen_update_buffer_resources(dxil_inst_name):
    fn_datastruct = 'AppendStructuredBuffer<float4> buffer;\n'
    dxil_op_name = 'bufferUpdateCounter'
    fn_body = "\n\tbuffer.Append(1);\n\treturn 0;\n}"
    return gen_resources_test(
        dxil_inst_name,
        fn_datastruct,
        dxil_op_name,
        fn_body)


def gen_cBufferload_legacy_resources(dxil_inst_name):
    fn_datastruct = 'cbuffer ConstantBuffer : register(b0) {\n\tuint4 buffer;};\n'
    dxil_op_name = 'cbufferLoadLegacy'
    return gen_resources_test(dxil_inst_name, fn_datastruct, dxil_op_name)


def gen_texture_load_resources(dxil_inst_name):
    fn_datastruct = 'Texture2D buffer;\n'
    dxil_op_name = 'textureLoad'
    fn_body = "\n\treturn buffer.Load(int3(0,0,0));\n}"
    return gen_resources_test(
        dxil_inst_name,
        fn_datastruct,
        dxil_op_name,
        fn_body)


def gen_texture_store_resources(dxil_inst_name):
    fn_datastruct = 'RWTexture2D<float4> buffer;\n'
    dxil_op_name = 'textureStore'
    return gen_resources_test(dxil_inst_name, fn_datastruct, dxil_op_name)


def gen_texture_store_sample_resources(dxil_inst_name):
    fn_datastruct = "RWTexture2DMS<float4,8> buffer;\n"
    dxil_op_name = "textureStoreSample"
    fn_sig = 'float4 fn(uint sampleSlice : S, uint2 pos2 : PP) : SV_Target {'
    fn_body = "\n\tfloat4 r = buffer.sample[sampleSlice][pos2];\n\tbuffer[pos2] = r;\n\treturn r;\n}"

    return gen_resources_test(
        dxil_inst_name,
        fn_datastruct,
        dxil_op_name,
        fn_body)


def gen_create_handle_from_binding(dxil_inst_name):
    fn_datastruct = 'StructuredBuffer<uint> buffer;\n'
    dxil_op_name = dxil_inst_name[:1].lower() + dxil_inst_name[1:]
    return gen_resources_test(dxil_inst_name, fn_datastruct, dxil_op_name)


def gen_get_dimensions(dxil_inst_name):
    fn_datastruct = 'RWBuffer<uint3> buffer;\n'
    dxil_op_name = "getDimensions"
    fn_body = '\n\tuint r = 0, d=0;\n\tbuffer.GetDimensions(d);\n\tr += d;\n\treturn r;\n}'
    return gen_resources_test(
        dxil_inst_name,
        fn_datastruct,
        dxil_op_name,
        fn_body)


def gen_resources():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    resource_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category == "Resources":
            scratchpad = os.path.join(
                scratchpad_path, dxil_inst.name + "_test.hlsl")
            payload = ""
            if dxil_inst.name in ["CreateHandle", "CBufferLoad", ]:
                # Note: These Resources are skipped
                continue
            elif dxil_inst.name.startswith("RawBuffer"):
                payload = gen_raw_buffer_resources(dxil_inst.name)
            elif dxil_inst.name == "BufferUpdateCounter":
                payload = gen_update_buffer_resources(dxil_inst.name)
            elif dxil_inst.name.startswith("Buffer"):
                payload = gen_reg_buffer_resources(dxil_inst.name)
            elif dxil_inst.name == "CBufferLoadLegacy":
                payload = gen_cBufferload_legacy_resources(dxil_inst.name)
            elif dxil_inst.name == "TextureLoad":
                payload = gen_texture_load_resources(dxil_inst.name)
            elif dxil_inst.name == "TextureStore":
                payload = gen_texture_store_resources(dxil_inst.name)
            elif dxil_inst.name == "TextureStoreSample":
                payload = gen_texture_store_sample_resources(dxil_inst.name)
            elif dxil_inst.name == "GetDimensions":
                payload = gen_get_dimensions(dxil_inst.name)
            dxc_command[1] = scratchpad
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_inst.name,
                resource_to_opcode,
                fail_list)

    dxil_inst_names = {"CreateHandleFromBinding": "6_8", "CreateHandle": "6_5"}
    for dxil_inst_name in dxil_inst_names:
        scratchpad = os.path.join(
            scratchpad_path,
            dxil_inst_name + "_test.hlsl")
        dxc_command_copy = dxc_command.copy()
        dxc_command_copy[1] = scratchpad
        dxc_command_copy[2] = f"-T ps_{dxil_inst_names[dxil_inst_name]}"
        dxc_command_copy.append("-E fn")
        payload = gen_create_handle_from_binding(dxil_inst_name)
        write_payload(scratchpad, payload)
        run_hlsl_test(
            dxc_command_copy,
            dxil_inst_name,
            resource_to_opcode,
            fail_list)

    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(resource_to_opcode)
    return resource_to_opcode


def gen_mesh_test(dxil_inst_name, dxil_op_name,
                  fn_sig, fn_body, fn_data_strs=""):

    fn_attr = '[numthreads(1, 1, 1)]\n[outputtopology("triangle")]\n[shader("mesh")]'
    fn_body = '\n\tSetMeshOutputCounts(1, 1);\n' + fn_body
    intrinsic_to_dxil_map[dxil_inst_name] = dxil_op_name
    payload = f"{fn_data_strs}{fn_attr}\n{fn_sig}{fn_body}"

    return payload


def gen_emit_indices(dxil_inst_name):
    dxil_op_name = "emitIndices"
    fn_body = '\n\tprimIndices[0] = uint3(1, 2, 3);\n}'
    fn_sig = 'void fn(out indices uint3 primIndices[1]) {'
    return gen_mesh_test(dxil_inst_name, dxil_op_name, fn_sig, fn_body)


def gen_store_vertex_output(dxil_inst_name):
    dxil_op_name = "storeVertexOutput"
    fn_data_strs = 'struct MeshPerVertex {\n\tfloat4 position : SV_Position;\n\tfloat color[4] : COLOR;\n};\n\n'
    fn_body = '\n\tMeshPerVertex ov;\n\tverts[0] = ov;\n}'
    fn_sig = 'void fn( out vertices MeshPerVertex verts[1]) {'
    return gen_mesh_test(
        dxil_inst_name,
        dxil_op_name,
        fn_sig,
        fn_body,
        fn_data_strs)


def gen_store_primitive_output(dxil_inst_name):
    dxil_op_name = "storePrimitiveOutput"
    fn_data_strs = 'struct MeshPerPrimitive {\n\tfloat normal : NORMAL;\n};\n\n'
    fn_body = '\n\tMeshPerPrimitive op;;\n\tprims[0] = op;\n}'
    fn_sig = 'void fn( out primitives MeshPerPrimitive prims[1]) {'
    return gen_mesh_test(
        dxil_inst_name,
        dxil_op_name,
        fn_sig,
        fn_body,
        fn_data_strs)


def gen_mesh_payload(dxil_inst_name):
    dxil_op_name = "getMeshPayload"
    fn_data_strs = 'RWBuffer<float> buffer;\nstruct MeshPayload {\n\tfloat normal;\n};\n\n'
    fn_body = '\n\tbuffer[0] = mpl.normal;\n}'
    fn_sig = 'void fn( in payload MeshPayload mpl) {'
    return gen_mesh_test(
        dxil_inst_name,
        dxil_op_name,
        fn_sig,
        fn_body,
        fn_data_strs)


def gen_mesh_shader_instr():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    mesh_shader_instr_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category == "Mesh shader instructions":
            scratchpad = os.path.join(
                scratchpad_path, dxil_inst.name + "_test.hlsl")
            payload = ''
            if dxil_inst.name in ["SetMeshOutputCounts"]:
                # covered via intrinsics
                continue
            if dxil_inst.name == "EmitIndices":
                payload = gen_emit_indices(dxil_inst.name)
            if dxil_inst.name == "StoreVertexOutput":
                payload = gen_store_vertex_output(dxil_inst.name)
            if dxil_inst.name == "StorePrimitiveOutput":
                payload = gen_store_primitive_output(dxil_inst.name)
            if dxil_inst.name == "GetMeshPayload":
                payload = gen_mesh_payload(dxil_inst.name)
            dxc_command[1] = scratchpad
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_inst.name,
                mesh_shader_instr_to_opcode,
                fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(mesh_shader_instr_to_opcode)
    return mesh_shader_instr_to_opcode


def gen_sample(dxil_inst_name):
    fn_attr = '[shader("pixel")]'
    fn_sig = "float4 fn(float2 a : A) : SV_Target {"
    fn_body = f'\n\treturn text1.{dxil_inst_name}(samp1, a'
    fn_datastruct = 'Texture2D<float4> text1 : register(t3);\n'

    if dxil_inst_name in ["Sample", "SampleBias", "SampleGrad", "SampleLevel"]:
        fn_datastruct += "SamplerState samp1 : register(s5);"
    elif dxil_inst_name in ["SampleCmp", "SampleCmpLevelZero",
                            "SampleCmpLevel", "SampleCmpBias", "SampleCmpGrad"]:
        fn_datastruct += "SamplerComparisonState samp1 : register(s5);"
        fn_body = '\n\tuint cmp = a.y + a.x;\n' + fn_body + ', cmp'

    if dxil_inst_name in ["SampleCmpLevel", "SampleLevel"]:
        fn_body = '\n\tfloat level = a.y;\n' + fn_body + ', level'

    if dxil_inst_name in ["SampleBias", 'SampleCmpBias']:
        fn_body = '\n\tfloat bias = a.y;\n' + fn_body + ', bias'
    if dxil_inst_name in ["SampleGrad", "SampleCmpGrad"]:
        fn_body = '\n\tfloat2 dx = a.yy, dy = a.yx;' + fn_body + ', dx, dy'
    fn_body += ");\n}"

    if dxil_inst_name == "Texture2DMSGetSamplePosition":
        fn_sig = "float4 fn(uint a : S) : SV_Target {"
        fn_datastruct = "RWTexture2DMS<float4,8> text1;\n"
        fn_body = '\n\treturn text1.GetSamplePosition(a).xyxy;\n}'

    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    intrinsic_to_dxil_map[dxil_inst_name] = dxil_inst_name[:1].lower(
    ) + dxil_inst_name[1:]

    return payload


def gen_resource_sample():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    resource_sample_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category in [
                "Resources - sample", "Comparison Samples"]:
            scratchpad = os.path.join(
                scratchpad_path, dxil_inst.name + "_test.hlsl")
            if dxil_inst.name in [
                "RenderTargetGetSamplePosition",
                    "RenderTargetGetSampleCount"]:
                # covered via intrinsics
                continue

            payload = gen_sample(dxil_inst.name)
            dxc_command[1] = scratchpad
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_inst.name,
                resource_sample_to_opcode,
                fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(resource_sample_to_opcode)
    return resource_sample_to_opcode


def gen_sampler_feedback_test(dxil_inst_name, scratchpad):
    fn_attr = '[shader("pixel")]'
    fn_sig = "void fn(float2 a : A) : SV_Target {"
    fn_body = f'\n\tfloat2 coords2D = float2(1, 2);\n\tfbMinMip.{dxil_inst_name}(tex2D, samp, coords2D'
    fn_datastruct = 'FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> fbMinMip;\nTexture2D<float> tex2D;\nSamplerState samp;\n'

    if dxil_inst_name == "WriteSamplerFeedbackBias":
        fn_body = '\n\tfloat bias = a.y;\n' + fn_body + ', bias'
    if dxil_inst_name == "WriteSamplerFeedbackLevel":
        fn_body = '\n\tfloat level = a.y;\n' + fn_body + ', level'
    if dxil_inst_name == "WriteSamplerFeedbackGrad":
        fn_body = '\n\tfloat2 dx = a.yy, dy = a.yx;' + fn_body + ', dx, dy'

    fn_body += ");\n}"

    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    intrinsic_to_dxil_map[dxil_inst_name] = dxil_inst_name[:1].lower(
    ) + dxil_inst_name[1:]
    return payload


def gen_sampler_feedback():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    sampler_feedback_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category == "Sampler Feedback":
            scratchpad = os.path.join(
                scratchpad_path, dxil_inst.name + "_test.hlsl")
            payload = gen_sampler_feedback_test(dxil_inst.name, scratchpad)
            dxc_command[1] = scratchpad
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_inst.name,
                sampler_feedback_to_opcode,
                fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(sampler_feedback_to_opcode)
    return sampler_feedback_to_opcode


def gen_geometry_shader_test(dxil_inst_name):
    fn_datastruct = 'struct GSOutPSIn {\n\tfloat4  clr : COLOR0;\n\tfloat4  pos : SV_Position;\n};\n\n'
    fn_attr = '[shader("geometry")]\n[maxvertexcount(3)]'
    fn_sig = 'void main(inout TriangleStream<GSOutPSIn> stream) {'
    fn_body = '\n\tGSOutPSIn gpsIn;\n\tgpsIn.clr = float4(1, 2, 3, 4);\n\tgpsIn.pos = float4(5, 6, 7, 8);'
    if dxil_inst_name == "CutStream":
        fn_body += '\n\tstream.RestartStrip();'
    fn_body += '\n\tstream.Append(gpsIn);'
    fn_body += "\n}"
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    intrinsic_to_dxil_map[dxil_inst_name] = dxil_inst_name[:1].lower(
    ) + dxil_inst_name[1:]
    return payload


def gen_geometry_shader_instr():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    geometry_instr_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category == "Geometry shader":
            scratchpad = os.path.join(
                scratchpad_path, dxil_inst.name + "_test.hlsl")
            if dxil_inst.name in ["GSInstanceID", "EmitThenCutStream"]:
                # Note: EmitThenCutStream not emitted in HLOperationLower.cpp
                # Note: GSInstanceID is covered by Semantics
                continue
            dxc_command[1] = scratchpad
            payload = gen_geometry_shader_test(dxil_inst.name)
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_inst.name,
                geometry_instr_to_opcode,
                fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(geometry_instr_to_opcode)
    return geometry_instr_to_opcode


def gen_hull(dxil_inst_name):
    fn_datastruct = 'struct HSPerPatchData {\n\tfloat edges[3] : SV_TessFactor;\n\tfloat inside : SV_InsideTessFactor;\n};\n'
    fn_datastruct += 'struct PSSceneIn {\n\tfloat2 tex : TEXCOORD0;\n};'
    fn_datastruct += "\nstruct HSPerVertexData {\n\tPSSceneIn v;\n};"
    if dxil_inst_name == "StorePatchConstant":
        fn_datastruct += '\nHSPerPatchData fn() {\n\tHSPerPatchData d;\n\treturn d;\n}\n'
    if dxil_inst_name == "LoadOutputControlPoint":
        fn_datastruct += '\nHSPerPatchData fn(OutputPatch<HSPerVertexData, 3> outpoints) {\n\tHSPerPatchData d;\n\td.edges[ 0 ] = outpoints[0].v.tex.x;\n\treturn d;\n}\n'
    fn_attr = '[shader("hull")]\n[domain("tri")]\n[partitioning("fractional_odd")]\n[outputtopology("triangle_cw")]\n[patchconstantfunc("fn")]\n[outputcontrolpoints(3)]'
    fn_sig = 'HSPerVertexData main(uint id : SV_OutputControlPointID,const InputPatch< PSSceneIn, 3 > points ) {'
    fn_body = '\n\tHSPerVertexData v;\n\tv.v = points[ id ];\n\treturn v;\n}'
    # Generate the payload
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    return payload


def gen_hull_shader_instr():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T hs_6_8",
        "-E main",
        "-enable-16bit-types",
        "-O0"
    ]
    hull_instr_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category in [
                "Domain and hull shader", "Hull shader"]:
            scratchpad = os.path.join(
                scratchpad_path, dxil_inst.name + "_test.hlsl")
            if dxil_inst.name in ["OutputControlPointID", "LoadPatchConstant"]:
                # Note: LoadPatchConstant covered by domain tests
                # Note: OutputControlPointID is covered by Semantics
                continue
            dxc_command[1] = scratchpad
            payload = gen_hull(dxil_inst.name)
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_inst.name,
                hull_instr_to_opcode,
                fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(hull_instr_to_opcode)
    return hull_instr_to_opcode


nodeTypes = {
    "EmptyNodeOutput": {
        "IsValid": [("i1", "v"), "nodeOutputIsValid"],
        "ThreadIncrementOutputCount": [("v", "i32"), 'incrementOutputCount'],
        "GroupIncrementOutputCount": [("v", "i32"), 'incrementOutputCount']
    },
    "EmptyNodeInput": {
        "Count": [("i32", "v"), "getInputRecordCount"]
    },
    "NodeOutput<T>": {
        "GetThreadNodeOutputRecords": [("ThreadNodeOutputRecords<T>", "i32"), "allocateNodeOutputRecords"],
        "GetGroupNodeOutputRecords": [("GroupNodeOutputRecords<T>", "i32"), "createNodeOutputHandle"]
    },
    "ThreadNodeOutputRecords<T>": {
        "OutputComplete": [("v", "v"), "outputComplete"],
    },
    "GroupNodeOutputRecords<T>": {
        "OutputComplete": [("v", "v"), "outputComplete"],
    },
    "RWDispatchNodeInputRecord<T>": {
        "FinishedCrossGroupSharing": [("i1", "v"), "finishedCrossGroupSharing"],
        # Technically getNodeRecordPtr,"
        "Get": [("T", "v"), "annotateNodeHandle"]
    },
    "DispatchNodeInputRecord<T>": {
        "Get": [("T", "v"), "getNodeRecordPtr"]
    },
    "ThreadNodeInputRecord<T>": {
        # Technically getNodeRecordPtr,"
        "Get": [("T", "v"), "createNodeInputRecordHandle"]
    },
    "RWThreadNodeInputRecord<T>": {
        # Technically getNodeRecordPtr,"
        "Get": [("T", "v"), "annotateNodeRecordHandle"]
    },
    "NodeOutputArray<T>": {
        # "[]" : [("NodeOutput<T>", "i32"), "indexNodeHandle"],
        "GetThreadNodeOutputRecords": [("ThreadNodeOutputRecords<T>", "i32"), "indexNodeHandle"],
        "GetGroupNodeOutputRecords": [("GroupNodeOutputRecords<T>", "i32"), "annotateNodeHandle"]
    }
}

node_class_initializers = {
    "ThreadNodeOutputRecords<T>": "GetThreadNodeOutputRecords",
    "GroupNodeOutputRecords<T>": "GetGroupNodeOutputRecords"
}

node_func_name_to_class_dict = {
    "ThreadIncrementOutputCount": ["EmptyNodeOutput"],
    "GroupIncrementOutputCount": ["EmptyNodeOutput"],
    "OutputComplete": [
        "ThreadNodeOutputRecords<T>",
        "GroupNodeOutputRecords<T>"],
    "Count": ["EmptyNodeInput"],
    "FinishedCrossGroupSharing": ["RWDispatchNodeInputRecord<T>"],
    "IsValid": ["EmptyNodeOutput"],
    "GetThreadNodeOutputRecords": ["NodeOutput<T>"],
    "GetGroupNodeOutputRecords": ["NodeOutput<T>"]}


def gen_node(dxil_name, class_name, fn_name, add_barrier=False):
    fn_datastruct = 'RWBuffer<uint> buffer;'
    fn_attr = '[Shader("node")]\n[NodeLaunch("thread")]\n[NodeIsProgramEntry]'
    if class_name in [
        "DispatchNodeInputRecord<T>",
            "RWDispatchNodeInputRecord<T>"]:
        fn_attr = '[Shader("node")]\n[NodeLaunch("broadcasting")]\n[numthreads(1,1,1)]\n[NodeDispatchGrid(1,1,1)]'
    if fn_name == "Count":
        fn_attr = '[Shader("node")]\n[NodeLaunch("coalescing")]\n[NodeIsProgramEntry]\n[numthreads(1,1,1)]'
    if fn_name == "FinishedCrossGroupSharing":
        fn_attr = '[Shader("node")]\n[NodeLaunch("broadcasting")]\n[NodeDispatchGrid(1,1,1)]\n[NumThreads(1,1,1)]'
    if fn_name in ["GetThreadNodeOutputRecords", "GetGroupNodeOutputRecords"]:
        fn_datastruct = ''

    param_name = f'node{"Output" if "Output" in class_name else ""}'
    param_name = f'node{"Input" if "Input" in class_name else ""}'
    templatized_class_name = class_name.replace("<T>", "<RECORD>")
    if class_name != templatized_class_name:
        fn_datastruct += '\nstruct '
        if fn_name == "FinishedCrossGroupSharing":
            fn_datastruct += '[NodeTrackRWInputSharing]'
        fn_datastruct += ' RECORD {\n\tuint value;\n};\n'
    fn_sig = f'void fn ({templatized_class_name} {param_name})'
    arr_syntax = '[0]' if class_name == "NodeOutputArray<T>" else ""
    fn_body = f'{param_name}{arr_syntax}.{fn_name}('
    if class_name in node_class_initializers:
        base_class = node_func_name_to_class_dict[node_class_initializers[class_name]][0]
        base_class = base_class.replace("<T>", "<RECORD>")
        fn_sig = f'void fn ({base_class} base{param_name})'
        fn_body = f'{templatized_class_name} {param_name} = base{param_name}{arr_syntax}.{node_class_initializers[class_name]}(1);\n\t{fn_body}'

    fn_sig += " {\n"

    fn_param_list = nodeTypes[class_name][fn_name][0]
    arg_length = len(fn_param_list)
    barrier_param = param_name
    if fn_name in ["GetThreadNodeOutputRecords", "GetGroupNodeOutputRecords"]:
        return_type = fn_param_list[0].replace("<T>", "<RECORD>")
        fn_body = f'\t{return_type} outrec = {fn_body}'
        barrier_param = 'outrec'
    elif fn_param_list[0] != 'v':
        fn_body = f'\tbuffer[0] = {fn_body}'
    else:
        fn_body = '\t' + fn_body
    for i in range(1, arg_length):
        type_name = get_valid_llvm_type(fn_param_list[i])
        if type_name == "void":
            continue
        type_val = int(1)
        if type_name == 'float' or type_name == 'double' or type_name == 'half':
            type_val = float(1.0)
        fn_body += f"{type_val}, "
    fn_body = fn_body.rstrip(", ") + ')'

    if fn_name == "Get":
        fn_body += ".value"

    fn_body += ";"
    if add_barrier:
        fn_body += f'\n\tBarrier({barrier_param}, 0);'
        intrinsic_to_dxil_map[dxil_name] = 'barrierByNodeRecordHandle'
    else:
        intrinsic_to_dxil_map[dxil_name] = nodeTypes[class_name][fn_name][1]
    fn_body += '\n}'

    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"

    return payload


def gen_calc_lod():
    fn_datastruct = 'Texture2D<float4> tex2d : register(t0, space0);'
    fn_datastruct += '\nSamplerState Samp : register(s0, space0);'
    fn_datastruct += '\nRWBuffer<float> buffer;'
    fn_attr = '[shader("node")]\n[NodeLaunch("broadcasting")]\n[NodeDispatchGrid(1, 1, 1)]\n[NumThreads(2,2,2)]'
    fn_sig = 'void fn(uint3 tid : SV_GroupThreadID) {'
    fn_body = '\n\tfloat2 uv = tid.xy;\n\tbuffer[0] = tex2d.CalculateLevelOfDetail(Samp, uv);\n}'
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"

    intrinsic_to_dxil_map['CalculateLevelOfDetail'] = 'calculateLOD'

    return payload


def gen_barrier_by_memory_handle():
    fn_datastruct = 'RWBuffer<float> buffer;\n'
    fn_attr = '[shader("node")]\n[NodeLaunch("broadcasting")]\n[NodeDispatchGrid(1, 1, 1)]\n[NumThreads(2,2,2)]'
    fn = 'void fn() {\n\tBarrier(buffer, 3);\n}'
    payload = f"{fn_datastruct}{fn_attr}\n{fn}"
    intrinsic_to_dxil_map['BarrierByMemoryHandle'] = 'barrierByMemoryHandle'
    return payload


def gen_create_handle_from_heap():
    fn_datastruct = 'RWBuffer<float> buffer;\n'
    fn_attr = '[shader("compute")]\n[numthreads(1, 1, 1)]'
    fn_sig = 'void fn( uint2 ID : SV_DispatchThreadID) {'
    fn_body = '\n\tBuffer<float> buf = ResourceDescriptorHeap[NonUniformResourceIndex(ID.x)];\n\tbuffer[0] = buf[0];\n}'
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"
    intrinsic_to_dxil_map['CreateHandleFromHeap'] = 'createHandleFromHeap'
    return payload


def gen_node_shader_instr():
    work_graph_to_fn_name_dict = {
        "IncrementOutputCount": [
            "ThreadIncrementOutputCount",
            "GroupIncrementOutputCount"],
        "OutputComplete": ["OutputComplete"],
        "GetInputRecordCount": ["Count"],
        "FinishedCrossGroupSharing": ["FinishedCrossGroupSharing"],
        "NodeOutputIsValid": ["IsValid"],
    }

    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    node_instr_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category in [
                "Work Graph intrinsics"]:
            if dxil_inst.name in ["GetRemainingRecursionLevels"]:
                # Note: GetRemainingRecursionLevels covered by intrinsics
                continue
            for fn_name in work_graph_to_fn_name_dict[dxil_inst.name]:
                for class_name in node_func_name_to_class_dict[fn_name]:
                    dxil_name = f'{class_name.split("<")[0]}_{fn_name}'
                    scratchpad = os.path.join(
                        scratchpad_path, dxil_name + "_test.hlsl")
                    dxc_command[1] = scratchpad
                    payload = gen_node(dxil_name, class_name, fn_name)
                    write_payload(scratchpad, payload)
                    run_hlsl_test(
                        dxc_command,
                        dxil_name,
                        node_instr_to_opcode,
                        fail_list)

    class_names = [
        "NodeOutput<T>",
        "DispatchNodeInputRecord<T>",
        "RWDispatchNodeInputRecord<T>",
        "RWThreadNodeInputRecord<T>",
        "ThreadNodeInputRecord<T>",
        "NodeOutputArray<T>"]
    for class_name in class_names:
        for fn_name in nodeTypes[class_name]:
            if fn_name == "FinishedCrossGroupSharing":
                continue
            dxil_name = f'{class_name.split("<")[0]}_{fn_name}'
            scratchpad = os.path.join(
                scratchpad_path, dxil_name + "_test.hlsl")
            dxc_command[1] = scratchpad
            payload = gen_node(dxil_name, class_name, fn_name)
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_name,
                node_instr_to_opcode,
                fail_list)

            dxil_name = f'barrier_{class_name.split("<")[0]}_{fn_name}'
            scratchpad = os.path.join(
                scratchpad_path, dxil_name + "_test.hlsl")
            dxc_command[1] = scratchpad
            payload = gen_node(dxil_name, class_name, fn_name, True)
            write_payload(scratchpad, payload)
            run_hlsl_test(
                dxc_command,
                dxil_name,
                node_instr_to_opcode,
                fail_list)

    dxil_names = {'CalculateLevelOfDetail': gen_calc_lod,
                  "BarrierByMemoryHandle": gen_barrier_by_memory_handle,
                  'CreateHandleFromHeap': gen_create_handle_from_heap}
    for dxil_name in dxil_names:
        scratchpad = os.path.join(scratchpad_path, dxil_name + "_test.hlsl")
        payload = dxil_names[dxil_name]()
        write_payload(scratchpad, payload)
        dxc_command[1] = scratchpad
        run_hlsl_test(dxc_command, dxil_name, node_instr_to_opcode, fail_list)

    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(node_instr_to_opcode)
    return node_instr_to_opcode


def gen_texture_gather_test(dxil_inst_name, fn_name):
    fn_attr = '[shader("pixel")]'
    fn_sig = 'uint16_t4 fn(float2 a : A) : SV_Target {\n'
    fn_datastruct = 'SamplerState g_samp : register(s5);\n'
    fn_body = f'return g_tex.{fn_name}(g_samp, a'

    if dxil_inst_name == "TextureGatherCmp":
        fn_datastruct = 'SamplerComparisonState g_samp;\n'
        fn_body += ', 0.0'

    if dxil_inst_name in ["TextureGather", "TextureGatherCmp"]:
        fn_datastruct += 'Texture2D g_tex;'
        fn_body += ', int2(1,1)'

    if dxil_inst_name == "TextureGatherRaw":
        fn_datastruct += 'Texture2D<uint16_t> g_tex : register(t1);'

    fn_body += ');\n}'

    intrinsic_to_dxil_map[fn_name] = dxil_inst_name[:1].lower() + \
        dxil_inst_name[1:]
    payload = f"{fn_datastruct}{fn_attr}\n{fn_sig}{fn_body}"

    return payload


def gen_texture_gather():

    texture_gather_to_fn_name_dict = {
        "TextureGather": [
            "Gather",
            "GatherRed",
            "GatherGreen",
            "GatherBlue",
            "GatherAlpha"],
        "TextureGatherCmp": [
            "GatherCmp",
            "GatherCmpRed",
            "GatherCmpGreen",
            "GatherCmpBlue",
            "GatherCmpAlpha"],
        "TextureGatherRaw": ["GatherRaw"],
    }

    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types",
        "-O0"
    ]
    texture_gather_instr_to_opcode = {}
    fail_list = []

    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op and dxil_inst.category == "Resources - gather":
            for fn_name in texture_gather_to_fn_name_dict[dxil_inst.name]:
                scratchpad = os.path.join(
                    scratchpad_path, fn_name + "_test.hlsl")
                dxc_command[1] = scratchpad
                payload = gen_texture_gather_test(dxil_inst.name, fn_name)
                write_payload(scratchpad, payload)
                run_hlsl_test(
                    dxc_command,
                    fn_name,
                    texture_gather_instr_to_opcode,
                    fail_list)
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    else:
        print_cli(texture_gather_instr_to_opcode)
    return texture_gather_instr_to_opcode

no_vulkan_equivalent = ['GetRenderTargetSampleCount', 'GetRenderTargetSamplePosition']
spirv_unimplemented = [ 'AddUint64', 'WaveMultiPrefixCountBits', 'EvaluateAttributeAtSample',
                       'EvaluateAttributeCentroid', 'EvaluateAttributeSnapped', 
                       'InterlockedCompareStoreFloatBitwise', 'InterlockedCompareExchangeFloatBitwise',
                       'QuadAny', 'QuadAll', 'ObjectToWorld', 'WorldToObject', 'Barrier', 'GetRemainingRecursionLevels']

spirv_broken = ['asdouble', 'asuint', 'and', 'or']

vulkan_pixel_shader = ['WaveGetLaneCount', 'WaveGetLaneIndex', 'InstanceIndex', 'PrimitiveIndex']

intersection_intrinsics = ['ObjectToWorld3x4', 'WorldToObject3x4', 'ObjectToWorld4x3', 'WorldToObject4x3', 'InstanceID',
                           'WorldRayDirection', 'WorldRayOrigin', 'ObjectRayOrigin', 'ObjectRayDirection',
                           'RayTMin', 'RayTCurrent', 'HitKind', 'RayFlags', 'DispatchRaysIndex', 'DispatchRaysDimensions',
                           'GeometryIndex'
                          ]

def gen_spirv_shader_instr():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "",
        "-enable-16bit-types",
        "-spirv",
        "-fspv-target-env=universal1.5",
        "-O0"
    ]
    deprecated_intrinsics = gen_deprecated_intrinsics()
    intrinsic_to_opcode = {}
    fail_list = []
    name_count = {}
    total_intrinsics = 0
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
                hl_op.name in hidden_intrinsics):
            continue
        if (hl_op.name in no_vulkan_equivalent or hl_op.name in spirv_unimplemented):
            continue
        # All hull intrinsics are listed as error: intrinsic function unimplemented
        if hl_op.name in hull_intrinsics or hl_op.name in spirv_broken:
            continue
        total_intrinsics = total_intrinsics + 1
        hl_op_name = get_unique_name(hl_op.name, name_count)
        dxc_intrinsic_spirv_run_helper(
                hl_op,
                dxc_command,
                scratchpad_path,
                hl_op_name,
                intrinsic_to_opcode,
                fail_list)
    print_cli(
        f"success %: {100.0 * ( (total_intrinsics - len(fail_list)) / total_intrinsics)}")
    if (len(fail_list) > 0):
        print_cli("FAILED:")
        print_cli(fail_list)
    return intrinsic_to_opcode



def main():
    """ Main function used for setting up the cli"""
    global isCli_print
    isCli_print = True
    args = parser.parse_args()
    functions = []
    if args.intrinsic_type_name:
        get_specifc_intrinsic_types(args.intrinsic_type_name)
        return 0
    if args.query_all_types:
        get_all_intrinsic_types()
        return 0
    if args.query_unique_types:
        get_all_types()
        return 0
    if args.gen_intrinsic_tests:
        run_dxc()
        return 0
    if args.gen_spirv_intrinsic_tests:
        gen_spirv_shader_instr()
        return 0
    if args.gen_semantic_tests:
        gen_semantics_hlsl()
        return 0
    if args.gen_rayquery_tests:
        gen_ray_query()
        return 0
    if args.gen_resource_tests:
        gen_resources()
        return 0
    if args.gen_resource_sample_tests:
        gen_resource_sample()
        return 0
    if args.gen_resource_gather_tests:
        gen_texture_gather()
        return 0
    if args.gen_sampler_feedback_tests:
        gen_sampler_feedback()
        return 0
    if args.gen_wavemat_tests:
        gen_wavematrix()
        return 0
    if args.gen_mesh_tests:
        gen_mesh_shader_instr()
        return 0
    if args.gen_geometry_tests:
        gen_geometry_shader_instr()
        return 0
    if args.gen_hull_tests:
        gen_hull_shader_instr()
        return 0
    if args.gen_node_tests:
        gen_node_shader_instr()
        return 0
    if args.query_all_dxil:
        query_dxil()
        return 0
    if args.build_dxc or args.rebuild_dxc or args.cov_build_dxc:
        build_dxc(args.rebuild_dxc, args.cov_build_dxc)
        return 0
    if args.get_hlsl_docs:
        printHLSLDoc()
        return 0
    if args.csv_doc_path:
        gen_csv(args.csv_doc_path)
        return 0

    return 0


if __name__ == '__main__':
    # pylint: disable-msg=C0103
    errcode = main()
    sys.exit(errcode)
    # pylint: enable-msg=C0103
