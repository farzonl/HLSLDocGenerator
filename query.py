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
from DirectXShaderCompiler.utils.hct.hctdb_instrhelp import get_db_hlsl
from DirectXShaderCompiler.utils.hct.hctdb_instrhelp import get_db_dxil
# pylint: enable=wrong-import-position

db_dxil = get_db_dxil()
db_hlsl = get_db_hlsl()
full_md_filepath = os.path.join(pathlib.Path().resolve(),
                                'win32/desktop-src/direct3dhlsl/dx-graphics-hlsl-intrinsic-functions.md')

deprecated_intrinsics = ["abort", "determinant", "sincos", "source_mark", 
                         "tex1D", "tex1Dbias", "tex1Dgrad", "tex1Dlod", "tex1Dproj", 
                         "tex2D", "tex2Dbias", "tex2Dgrad", "tex2Dlod", "tex2Dproj", 
                         "tex3D", "tex3Dbias", "tex3Dgrad", "tex3Dlod", "tex3Dproj", 
                         "texCUBE", "texCUBEbias", "texCUBEgrad", "texCUBElod", 
                         "texCUBEproj", "transpose"]

pixel_intrinsics = ["GetRenderTargetSampleCount", "GetRenderTargetSamplePosition", "clip"]

any_hit_intrinsics = ["IgnoreHit", "AcceptHitAndEndSearch"]

node_intrinsics = ["GetRemainingRecursionLevels"]

hidden_intrinsics = ["CreateResourceFromHeap", "AllocateRayQuery"]

mesh_intrinsics = ["SetMeshOutputCounts"]

amplification_intrinsics = ["DispatchMesh"]

hull_intrinsics = ['Process2DQuadTessFactorsAvg', 'Process2DQuadTessFactorsMax', 'Process2DQuadTessFactorsMin', 
                    'ProcessIsolineTessFactors', 'ProcessQuadTessFactorsAvg', 'ProcessQuadTessFactorsMax', 
                    'ProcessQuadTessFactorsMin', 'ProcessTriTessFactorsAvg', 'ProcessTriTessFactorsMax', 'ProcessTriTessFactorsMin']

const_intrinsitcs = ["Barrier"]

no_export = ["TraceRay","EvaluateAttributeAtSample", "EvaluateAttributeCentroid", "EvaluateAttributeSnapped", "GetAttributeAtVertex"]

DXC_PATH = os.path.join(pathlib.Path().resolve(),"DXC_Debug_BUILD/bin/dxc")

isCli_print = False

def print_cli(*args, **kwargs):
    global isCli_print
    if isCli_print:
        print( " ".join(map(str,args)), **kwargs)

def md_table_to_dict(md_file):
    html_content = pypandoc.convert_file(
        full_md_filepath,
        "html",
        format="markdown",
        extra_args=["--standalone"])

    # print(html_content)

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')

    # Extracting headers
    headers = [header.get_text() for header in table.find_all('th')]
    name_index = headers.index('Name')
    desc_index = headers.index('Description')
    shader_index = headers.index('Minimum shader model')

    # Extracting data
    data = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        name = cells[name_index].get_text()
        description = cells[desc_index].get_text()
        # shaders only went from 1-5 before DXC
        # after is 6.0-6.8
        # this modification lets us drop the superscrpt
        shader_version = cells[shader_index].get_text()
        if(len(shader_version) > 1 and shader_version[1] != '.'):
            shader_version = cells[shader_index].get_text()[0]
        data[name] = [description, shader_version]

    return data


def extract_opcode(line):
    match = re.search(r'@dx\.op\..*\(i32\s*(?P<int_opcode>\d+)', line)
    if match:
        return int(match.group('int_opcode'))
    else:
        return None


def get_valid_type(type_name):
    type_name = type_name.strip()
    if (type_name == "numeric<4>" or type_name == "numeric<c>"
        or type_name == "numeric<>" or type_name == "float_like<c>"
        or type_name == "float<>" or type_name == "float<4>"
        or type_name == "any<>" or type_name == "float_like<4>"
        or type_name == "float_like<>" or type_name == "any_float<>"
        or type_name == "any_sampler" or type_name == "numeric<c2>"
        or type_name == "numeric<r>"):
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
        or type_name == "numeric32_only<>" or type_name == "any_int<>"
        or type_name == "sint16or32_only<4>" or type_name == "any_int16or32<4>"):
        return "int4"
    if type_name == "p32i8":
        return "int8_t4_packed"
    if type_name == "p32u8":
        return "uint8_t4_packed"
    if type_name == "int<3>":
        return "int3"
    if type_name == "int<2>":
        return "int2"
    if (type_name == "int16_t<4>" or type_name == "numeric16_only<>"
        or type_name == "int16_t<>"):
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
    if (type_name == "numeric" or type_name == "float_like" 
        or type_name == "float<1>" or type_name == "float32_only"):
        return "float"
    if type_name == "u64":
        return "uint64_t"
    if  type_name == "int64_only":
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
    if (type_name == "numeric<r2@c2>" or type_name == "numeric<c2@r2>"
        or type_name == "numeric<r@c>" or type_name == "numeric<c@r>"
        or type_name == "numeric<c@c2>" or type_name == "numeric<r@r2>"
        or type_name == "numeric<r@c2>" or type_name == "numeric<c@r2>"):
        return "float4x4"
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

def generate_interlocked(scratchpad, func_name, params):
    code_body = f"RWStructuredBuffer<{get_valid_type(params[1].type_name)}> buffer : register(u0);\n[numthreads(1, 1, 1)]\n"
    return_type = params[0].type_name
    code_body += f"export {get_valid_type(return_type)} fn(uint3 dispatchThreadID : SV_DispatchThreadID, "
    arg_length = len(params) -1
    for i in range(1, arg_length):
        code_body += f"{get_valid_type(params[i].type_name)} p{i}, "
    code_body = code_body.rstrip(", ") + ") {\nint index = dispatchThreadID.x;"

    # Define the function call
    func_call = f"return {func_name}(buffer[index], "
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

     # Generate the payload
    payload = f"{code_body}\n    {func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)

def generate_node(scratchpad, func_name, params):
    return_type = params[0].type_name
    arg_length = len(params)
    buffer = f"RWBuffer<{get_valid_type(return_type)}> buf0;\n"
    fn_header = '[shader("node")]\n[NodeDispatchGrid(1, 1, 1)]\n[numthreads(1, 1, 1)]\nvoid fn() {\n'
    
    # Define the function call
    func_call = f"buf0[0] = {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{buffer}\n{fn_header}\n{func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)

def generate_anyhit(scratchpad, func_name, params):
    raypayload = 'struct [raypayload] RayPayload\n{\n\tfloat4 color : write(caller) : read(anyhit);\n\tfloat distance : write(caller) : read(anyhit);\n};\n'
    attributes = 'struct Attributes {\n\tfloat3 barycentrics;\n\tuint primitiveIndex;\n};\n'
    shader_header ='[shader("anyhit")]\nexport void fn(inout RayPayload payload, in Attributes attributes) {\n'
    arg_length = len(params)

    # Define the function call
    func_call = f"return {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{raypayload}\n{attributes}\n{shader_header}\t{func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)

def generate_mesh(scratchpad, func_name, params):
    fn_attr = '[numthreads(1, 1, 1)]\n[outputtopology("triangle")]\n[shader("mesh")]'
    fn_sig =  'void fn(in uint gi : SV_GroupIndex, in uint vi : SV_ViewID) {\n'
    arg_length = len(params)

    func_call = f"{func_name}("
    for i in range(1, arg_length):
        type_name = get_valid_type(params[i].type_name)
        type_val = int(1) 
        if type_name == 'float' or type_name == 'double' or type_name == 'half':
            type_val = float(1.0)
        func_call += f"{type_val}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{fn_attr}\n{fn_sig}\n{func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)


def generate_amplification(scratchpad, func_name, params):
    fn_attr = '[numthreads(1, 1, 1)]\n[shader("amplification")]'
    fn_sig =  'export void fn(uint gtid : SV_GroupIndex) {\n'
    rayPayload = "struct RayPayload\n{\n\tfloat4 color;\n\tfloat distance;\n};"
    rayPayloadInserted = False
    arg_length = len(params)
    arg_list = ""
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name)
        if not rayPayloadInserted and arg_type == "RayPayload":
            arg_list = rayPayload +"\n" + arg_list
        arg_list += f"{arg_type} p{i};\n"
    
    # Define the function call
    func_call = f"{func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    # Generate the payload
    payload = f"{fn_attr}\n{fn_sig}\n{arg_list}\n{func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)

def generate_pixel(scratchpad, func_name, params):
    fn_attr = '[numthreads(1, 1, 1)]\n[shader("pixel")]'
    return_type = params[0].type_name
    signature = f"{get_valid_type(return_type)} fn() : SV_Target"
    signature += "{\n"
    arg_length = len(params)
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name)    
        signature += f"{arg_type} p{i};\n"

    # Define the function call
    func_call = f"return {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{fn_attr}\n{signature}\n    {func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)

def generate_hull(scratchpad, func_name, params):
    hs_per_patch_data = 'struct HSPerPatchData\n{\n\tfloat edges[ 3 ] : SV_TessFactor;\n\tfloat inside : SV_InsideTessFactor;\n};\n'
    ps_scene_in = 'struct PSSceneIn {\n\t float4 pos : SV_Position;\n\tfloat2 tex : TEXCOORD0;\n\tfloat3 norm : NORMAL;\n};\n'
    hs_per_vertex_data = 'struct HSPerVertexData {\n\tPSSceneIn v;\n};\n'
    data_structures = f'{hs_per_patch_data}{ps_scene_in}{hs_per_vertex_data}'
    main_fn_attr = '[domain("tri")]\n[outputtopology("triangle_cw")]\n[patchconstantfunc("fn")]\n[outputcontrolpoints(3)]\n[partitioning("pow2")]'
    main_fn_sig  = 'HSPerVertexData main( const uint id : SV_OutputControlPointID,const InputPatch< PSSceneIn, 3 > points ) {'
    main_fn_body = 'HSPerVertexData v;\nv.v = points[ id ];\nreturn v;\n}'

    fn_sig = 'export HSPerPatchData fn( const InputPatch< PSSceneIn, 3 > points ) {\n\tHSPerPatchData d;\n\tfloat4 edgeFactor;\n\tfloat2 insideFactor;'
    fn_body_end = '\td.edges[0]=edgeFactor.x;\n\td.edges[1]=edgeFactor.y;\n\td.edges[2]=edgeFactor.z + edgeFactor.w;\n\td.inside = insideFactor.x + insideFactor.y;\n\treturn d;\n}\n'

    arg_length = len(params)
    fn_args = ""
    for i in range(1, arg_length):
        arg_type = get_valid_type(params[i].type_name)    
        fn_args += f"{arg_type} p{i};\n"
    
    func_call = f"{func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n"
    
    fn_test_str = f"{fn_sig}{fn_args}{func_call}{fn_body_end}"
    payload = f"{data_structures}\n{fn_test_str}\n{main_fn_attr}\n{main_fn_sig}\n{main_fn_body}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)


def generate_scratch_file(scratchpad, func_name, params):
    # Define the function signature
    if func_name.startswith("Interlocked"):
        generate_interlocked(scratchpad, func_name, params)
        return
    if func_name in any_hit_intrinsics:
        generate_anyhit(scratchpad, func_name, params)
        return
    if func_name in node_intrinsics:
        generate_node(scratchpad, func_name, params)
        return
    if func_name in mesh_intrinsics:
        generate_mesh(scratchpad, func_name, params)
        return 
    if func_name in amplification_intrinsics:
        generate_amplification(scratchpad, func_name, params)
        return 
    if func_name in pixel_intrinsics:
        generate_pixel(scratchpad, func_name, params)
        return 
    if func_name in hull_intrinsics:
        generate_hull(scratchpad, func_name, params)
        return

    rayPayload = "struct RayPayload\n{\n\tfloat4 color;\n\tfloat distance;\n};"
    rayPayloadInserted = False
    return_type = params[0].type_name
    if(func_name in no_export ):
        signature = f"{get_valid_type(return_type)} fn("
    else:
        signature = f"export {get_valid_type(return_type)} fn("
    arg_length = len(params)
    if func_name == "printf":
        arg_length = arg_length -1
    if func_name in const_intrinsitcs:
        signature = signature + ") {\n"
        for i in range(1, arg_length):
            type_name = params[i].type_name
            type_val = int(1) 
            if type_name == 'float' or type_name == 'double' or type_name == 'half':
                type_val = float(1.0)
            arg_type = get_valid_type(type_name)
            signature += f"\t{arg_type} p{i} = {type_val};\n"
    else:
        for i in range(1, arg_length):
            arg_type = get_valid_type(params[i].type_name)
            if not rayPayloadInserted and arg_type == "RayPayload":
                signature = rayPayload +"\n" + signature
            signature += f"{arg_type} p{i}, "
        signature = signature.rstrip(", ") + ") {"

    # Define the function call
    func_call = f"return {func_name}("
    for i in range(1, arg_length):
        func_call += f"p{i}, "
    func_call = func_call.rstrip(", ") + ");\n}"

    # Generate the payload
    payload = f"{signature}\n    {func_call}"
    # Write the payload to a file
    with open(scratchpad, "w") as file:
        file.write(payload)

def printHLSLDoc():
    hlsl_intrinsic_doc_dict = md_table_to_dict(full_md_filepath)
    intrinsics_no_docs = []
    hlsl_intrinsics_count = 0
    for hl_op in db_hlsl.intrinsics:
      if (hl_op.ns != "Intrinsics" or hl_op.name in hidden_intrinsics):
            continue
      hlsl_intrinsics_count = hlsl_intrinsics_count + 1
      if hl_op.name in hlsl_intrinsic_doc_dict:
          print(f'{hl_op.name} args {len(hl_op.params)} docs: {hlsl_intrinsic_doc_dict[hl_op.name]}')
      else:
          intrinsics_no_docs.append(hl_op.name)
    print("no docs:")
    print(intrinsics_no_docs)
    print(f'percentage without docs: {100.0 * (len(intrinsics_no_docs)/ hlsl_intrinsics_count)}%')
        
          

def build_dxc(rebuild: bool = False):
    if not rebuild and os.path.exists(DXC_PATH):
        print("Build Succeeded!")
        return
    
    if rebuild:
        try:
            shutil.rmtree(os.path.join(pathlib.Path().resolve(),"DXC_Debug_BUILD"))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    cmake_dxc_build_command = [
        "cmake",
        "-S DirectXShaderCompiler", 
        "-B DXC_Debug_BUILD", 
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DLLVM_ENABLE_LLD=ON",
        "-G Ninja",
        "-C DirectXShaderCompiler/cmake/caches/PredefinedParams.cmake"
    ]
    ninja_dxc_build_command = ["ninja", "-C", "DXC_Debug_BUILD"]

    result = subprocess.run(cmake_dxc_build_command, capture_output=True, text=True)
    if result.returncode == 0:
        print("CMake Succeeded!")
    else:
        print("CMake Failed!")
        print(result.stderr)
        #printverbose(result.stdout)
        return
    
    result = subprocess.run(ninja_dxc_build_command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Build Succeeded!")
    else:
        print("Build Failed!")
        print(result.stderr)
        #printverbose(result.stdout)

# Run dxc
def run_dxc():
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    dxc_command = [
        DXC_PATH,
        "<scratchpad_placholder>",
        "-T lib_6_8",
        "-enable-16bit-types"
    ]

    intrinsic_to_opcode = {}
    fail_list = []
    name_count = {}
    total_intrinsics = 0
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
            hl_op.name in hidden_intrinsics):
            continue
        if hl_op.name in name_count:
            name_count[hl_op.name] = name_count[hl_op.name] + 1
        else:
            name_count[hl_op.name] = 0 
        total_intrinsics = total_intrinsics + 1
        hl_op_name = f"{hl_op.name}{'' if name_count[hl_op.name] == 0 else name_count[hl_op.name]}"
        scratchpad = os.path.join(scratchpad_path,  hl_op_name+"_test.hlsl")
        dxc_command[1] = scratchpad
        if hl_op.name in hull_intrinsics:
            dxc_command[2] = "-T hs_6_8"
        else:
            dxc_command[2] = "-T lib_6_8"
        generate_scratch_file(scratchpad, hl_op.name, hl_op.params)
        result = subprocess.run(dxc_command, capture_output=True, text=True)
        if result.returncode == 0:
            opcode = extract_opcode(result.stdout)
            if (opcode == None):
                intrinsic_to_opcode[hl_op_name] = -1
            else:
                opcode = int(opcode)
                intrinsic_to_opcode[hl_op_name] = opcode
        else:
            fail_list.append(hl_op_name)
    
    print_cli(f"success %: {100.0 * ( (total_intrinsics - len(fail_list)) / total_intrinsics)}")
    print_cli("FAILED:")
    print_cli(fail_list)
    return intrinsic_to_opcode

def formatShaderModel(smodel):
    return "{}.{}".format(smodel[0], smodel[1])

def query_dxil():
    opcode_to_dxil_docs = {}
    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op:
            opcode_to_dxil_docs[dxil_inst.dxil_opid] = [dxil_inst.dxil_op, formatShaderModel(dxil_inst.shader_model), dxil_inst.shader_stages, dxil_inst.doc]
            print_cli(dxil_inst.dxil_op, dxil_inst.dxil_opid, formatShaderModel(dxil_inst.shader_model), dxil_inst.shader_stages, dxil_inst.doc)
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

def get_specifc_intrinsic_types(func_name : str):
    type_set = set()
    for hl_op in db_hlsl.intrinsics:
        if(func_name != hl_op.name):
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
    if intrinsic_name and intrinsic_name[-1].isdigit():
        intrinsic_name = intrinsic_name[:-1]
    return hlsl_intrinsic_doc_dict[intrinsic_name] if intrinsic_name in hlsl_intrinsic_doc_dict else ["",""]

def gen_csv_data_base(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_intrinsic_doc_dict):
    data = [
    ['HLSL Intrinsic', 'DXIL Opcode', 'DXIL OpName', "Shader Model", "Shader Model DXC", "Shader Stages", "DXIL Docs", "HLSL Docs"]
    ]
    
    for key, value in hlsl_to_dxil_op.items():
        dxil_docs = dxil_op_to_docs[value] if value != -1 else ["","","",""]
        hlsl_docs = hlsl_docs_helper(hlsl_intrinsic_doc_dict, key)
        row = [key, value, dxil_docs[0], hlsl_shader_model_helper(dxil_docs[1],hlsl_docs[1]), dxc_shader_model_helper(dxil_docs[1]), dxil_docs[2], dxil_docs[3], hlsl_docs[0]]
        data.append(row)
    return data

def gen_csv_data():
    hlsl_to_dxil_op = run_dxc()
    dxil_op_to_docs = query_dxil()
    hlsl_intrinsic_doc_dict = md_table_to_dict(full_md_filepath)
    return gen_csv_data_base(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_intrinsic_doc_dict)

def gen_csv(csv_file_path : str):
    data = gen_csv_data()
    # Open the file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Example usage:
# line = "%9 = call float @dx.op.dot4.f32(i32 56, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8)  ; Dot4(ax,ay,az,aw,bx,by,bz,bw)"
# opcode = extract_opcode(line)
# print(opcode)  # Output: 56

def get_missing_opcodes_from_docs(hlsl_to_dxil_op, dxil_op_to_docs):
    copied_dxil_op_to_docs = dxil_op_to_docs.copy()
    for key, value in hlsl_to_dxil_op.items():
        if value in copied_dxil_op_to_docs:
            del copied_dxil_op_to_docs[value]
    return copied_dxil_op_to_docs

def gen_remaining_opcode_data(hlsl_to_dxil_op, dxil_op_to_docs):
    opcode_count = len(dxil_op_to_docs)
    rem_opcodes = get_missing_opcodes_from_docs(hlsl_to_dxil_op, dxil_op_to_docs)
    data = [
    ['DXIL Opcode', 'DXIL OpName', "Shader Model DXC", "Shader Stages", "DXIL Docs"]
    ]
    for key, value in rem_opcodes.items():
        row = [key, value[0], dxc_shader_model_helper(value[1]), value[2], value[3]]
        data.append(row)
    return data
    
    
def main():
    """ Main function used for setting up the cli"""
    global isCli_print
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
    if args.gen_dxc_tests:
        isCli_print = True
        run_dxc()
        return 0
    if args.query_all_dxil:
        isCli_print = True
        query_dxil()
        return 0
    if args.build_dxc:
        build_dxc()
        return 0
    if args.rebuild_dxc:
        build_dxc(True)
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