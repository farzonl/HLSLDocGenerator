import os
import requests
import socket
import pathlib
import pickle
import re
from urllib3.exceptions import MaxRetryError, NameResolutionError
from query import load_dict, run_dxc, db_hlsl, gen_deprecated_intrinsics, hidden_intrinsics, query_dxil, gen_spirv_shader_instr, no_direct_x_support, no_spir_v_support, special_spirv_example_code_intrinsics
from query import  md_table_to_dict, full_md_filepath, query_dxil, load_dict

from llvm_git_graph import llvm_hlsl_completed_intrinsics, llvm_dxil_completed_ops
from spirv_doc_fetch import parse_spirv_spec, spirv_vulkan3_base_url, parse_spirv_vulkan_man_page
from utils import ApiKeys
import time
import traceback


# GitHub repository and personal access token
REPO_OWNER = 'llvm'
REPO_NAME = 'llvm-project'
GITHUB_TOKEN = ''
DEBUG_PRINT = False

deprecated_intrinsics = []

succesfully_created_issues = {}

successfully_created_intrinsics = [
"AllMemoryBarrier",
"AllMemoryBarrierWithGroupSync",
"asdouble",
"asfloat",
"asint",
"asuint",
"asuint",
"atan2",
"clip",
"countbits",
"cross",
"D3DCOLORtoUBYTE4",
"ddx",
"ddx_coarse",
"ddx_fine",
"ddy",
"ddy_coarse",
"ddy_fine",
"degrees",
"DeviceMemoryBarrier",
"DeviceMemoryBarrierWithGroupSync",
"distance",
"dst",
"EvaluateAttributeCentroid",
"EvaluateAttributeAtSample",
"EvaluateAttributeSnapped",
"f16tof32",
"f32tof16",
"faceforward",
"firstbithigh",
"firstbitlow",
"fma",
"fmod",
"frexp",
"fwidth",
"GetRenderTargetSampleCount",
"GetRenderTargetSamplePosition",
"GroupMemoryBarrier",
"GroupMemoryBarrierWithGroupSync",
"InterlockedAdd",
"InterlockedAnd",
"InterlockedCompareExchange",
"InterlockedCompareStore",
"InterlockedExchange",
"InterlockedMax",
"InterlockedMin",
"InterlockedOr",
"InterlockedXor",
"isfinite",
"isnan",
"ldexp",
"length",
"lit",
"modf",
"msad4",
"mul",
"normalize",
"printf",
"Process2DQuadTessFactorsAvg",
"Process2DQuadTessFactorsMax",
"Process2DQuadTessFactorsMin",
"ProcessIsolineTessFactors",
"ProcessQuadTessFactorsAvg",
"ProcessQuadTessFactorsMax",
"ProcessQuadTessFactorsMin",
"ProcessTriTessFactorsAvg",
"ProcessTriTessFactorsMax",
"ProcessTriTessFactorsMin",
"radians",
"reflect",
"refract",
"saturate",
"sign",
"smoothstep",
"step",
"WaveGetLaneCount",
"WaveIsFirstLane",
"WaveActiveAnyTrue",
"WaveActiveAllTrue",
"WaveActiveBallot",
"WaveReadLaneAt",
"WaveReadLaneFirst",
"WaveActiveAllEqual",
"WaveActiveBitAnd",
"WaveActiveBitOr",
"WaveActiveBitXor",
"WaveActiveMax",
"WaveActiveMin",
"WaveActiveProduct",
"WaveActiveSum",
"WavePrefixCountBits",
"WavePrefixSum",
"WavePrefixProduct",
"QuadReadLaneAt",
"QuadReadAcrossDiagonal",
"QuadReadAcrossX",
"QuadReadAcrossY",
"TraceRay",
"ReportHit",
"CallShader",
"IgnoreHit",
"AcceptHitAndEndSearch",
"DispatchRaysIndex",
"DispatchRaysDimensions",
"WorldRayOrigin",
"WorldRayDirection",
"ObjectRayOrigin",
"ObjectRayDirection",
"RayTMin",
"RayTCurrent",
"PrimitiveIndex",
"InstanceIndex",
"HitKind",
"RayFlags",
"ObjectToWorld3x4",
"WorldToObject3x4",
"ObjectToWorld4x3",
"WorldToObject4x3",
'asfloat16',
'asint16',
'asuint16',
'GetAttributeAtVertex',
'InterlockedCompareStoreFloatBitwise',
'InterlockedCompareExchangeFloatBitwise',
'CheckAccessFullyMapped',
'AddUint64',
'NonUniformResourceIndex',
'WaveMatch',
'WaveMultiPrefixBitAnd',
'WaveMultiPrefixBitOr',
'WaveMultiPrefixBitXor',
'WaveMultiPrefixCountBits',
'WaveMultiPrefixProduct',
'WaveMultiPrefixSum',
'QuadAny',
'QuadAll',
'InstanceID',
'GeometryIndex',
'ObjectToWorld',
'WorldToObject',
'dot4add_u8packed',
'dot4add_i8packed',
'dot2add',
'unpack_s8s16',
'unpack_u8u16',
'unpack_s8s32',
'unpack_u8u32',
'pack_s8',
'pack_u8',
'pack_clamp_s8',
'pack_clamp_u8',
'SetMeshOutputCounts',
'DispatchMesh',
'IsHelperLane',
'select',
'Barrier',
'GetRemainingRecursionLevels'
]

MD_PATH_PREFIX = 'https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/'

def filter_escapes(str):
    return str.replace('\\_', '_')

def replace_md_links_with_hyperlinks(markdown_content, md_file_parent_dir):
    # Define a regex pattern to match Markdown links
    pattern = r'\[([^\]]+)\]\(([^\)]+)\.md\)'

    # Replace Markdown links with hyperlinks
    def replace_link(match):
        title = match.group(1)
        filename = match.group(2)
        return f'[{title}]({MD_PATH_PREFIX}/{md_file_parent_dir}/{filename}.md)'

    # Perform the replacement
    replaced_content = re.sub(pattern, replace_link, markdown_content)
    return replaced_content

def save_issue_numbers(issue_numbers):
    """Saves issue numbers to a pickled file."""
    if len(issue_numbers) == 0:
        return
    issue_file_path = os.path.join(pathlib.Path().resolve(), 'scratch/issue_numbers.pkl')
    with open(issue_file_path, 'wb') as f:
        pickle.dump(issue_numbers, f)

def create_github_issue(title, body, labels):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "labels": labels
    }
    if DEBUG_PRINT:
        print(data)
        return None
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        if response.status_code == 201:
            print(f"Successfully created issue: {title}")
            return response.json().get('number')
    except socket.gaierror as e:
        print(f"Socket error: {e}")
    except NameResolutionError as e:
        print(f"Name resolution error: {e}")
    except MaxRetryError as e:
        print(f"Max retries exceeded: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    
    print(f"Failed to create issue: {title}")
    print(f"Response: {response.content}")
    return None

def create_github_issue_helper(intrinsic_name, body, hlsl_to_dxil_op, dxil_op_to_docs, 
                               hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,
                               hlsl_to_spirv_op, spirv_op_to_docs):
    if intrinsic_name in hlsl_ignore_intrinsics:
        return
    
    labels = ['HLSL', 'metabug', 'bot:HLSL']
    requirements_body = f'- [ ] Implement the `{intrinsic_name}`  api in `hlsl_intrinsics.h`\n'
    requirements_body += f'- [ ] If a clang builtin is needed add sema checks for `{intrinsic_name}` to `CheckHLSLBuiltinFunctionCall` in `SemaHLSL.cpp`\n'
    requirements_body += f'- [ ] If codegen is needed, add codegen for `{intrinsic_name}` to `EmitHLSLBuiltinExpr` in `CGBuiltin.cpp`\n'
    requirements_body += f'- [ ] Add codegen tests to `clang/test/CodeGenHLSL/builtins/{intrinsic_name}.hlsl`\n'
    requirements_body += f'- [ ] Add sema tests to `clang/test/SemaHLSL/BuiltIns/{intrinsic_name}-errors.hlsl`\n'
    dxil_op =  hlsl_to_dxil_op.get(intrinsic_name, -1)
    doc_body = '## DirectX\n\n'
    if dxil_op != -1:
        dxil_docs = dxil_op_to_docs[dxil_op]
        requirements_body += f'- [ ] Create the `int_dx_{intrinsic_name}` intrinsic in `IntrinsicsDirectX.td`\n'
        requirements_body += f'- [ ] Create the `DXILOpMapping` of `int_dx_{intrinsic_name}` to  `{dxil_op}` in `DXIL.td`\n'
        requirements_body += f'- [ ] Create the  `{intrinsic_name}.ll` and `{intrinsic_name}_errors.ll` tests in `llvm/test/CodeGen/DirectX/`\n'

        dxil_doc_body =    '| DXIL Opcode | DXIL OpName | Shader Model | Shader Stages |'
        dxil_doc_body += '\n| ----------- | ----------- | ------------ | ------------- |'
        dxil_doc_body += f'\n| {dxil_op} | {dxil_docs[0]} | {dxil_docs[1]} | {dxil_docs[3]} |' 
        doc_body += dxil_doc_body + '\n'
        labels.append('backend:DirectX')
    else:
        if intrinsic_name in no_direct_x_support:
            doc_body += f"There is no support for `{intrinsic_name}` when targeting DirectX.\n"
        else:
            doc_body += f"There were no DXIL opcodes found for `{intrinsic_name}`.\n"
    
    spirv_op =  hlsl_to_spirv_op.get(intrinsic_name, -1)
    doc_body += '\n## SPIR-V\n\n'
    if spirv_op != -1:
        spirv_man_url = f'{spirv_vulkan3_base_url}{spirv_op}.html'
        if spirv_op_to_docs[spirv_op] == spirv_man_url:
            spirv_doc = parse_spirv_vulkan_man_page(spirv_man_url)
        else:
            spirv_doc = spirv_op_to_docs[spirv_op]
        requirements_body += f'- [ ] Create the `int_spv_{intrinsic_name}` intrinsic in `IntrinsicsSPIRV.td`\n'
        requirements_body += f'- [ ] In SPIRVInstructionSelector.cpp create the `{intrinsic_name}` lowering and map  it to `int_spv_{intrinsic_name}` in `SPIRVInstructionSelector::selectIntrinsic`.\n'
        requirements_body += f'- [ ] Create SPIR-V backend test case in `llvm/test/CodeGen/SPIRV/hlsl-intrinsics/{intrinsic_name}.ll`\n'
        doc_body += spirv_doc + '\n'
        labels.append('backend:SPIR-V')
    else:
        if intrinsic_name in no_spir_v_support:
            doc_body += f"There is no support for `{intrinsic_name}` when targeting SPIR-V.\n"
        else:
            doc_body += f"There were no SPIR-V opcodes found for `{intrinsic_name}`.\n"
    
    if body != '':
        body = f'## HLSL:\n\n{body}'

    test_case_body = f'## Test Case(s)\n\n {gen_test_case_body(intrinsic_name, hlsl_intrinsics_test_cases)}'
    body = f'{requirements_body}\n{doc_body}\n{test_case_body}\n{body}' 
    title = f"Implement the `{intrinsic_name}` HLSL Function"
    
    issue_number = create_github_issue(title, body, labels)
    if issue_number:
        succesfully_created_issues[intrinsic_name] = issue_number
    else:
        save_issue_numbers(succesfully_created_issues)
    # waiting so github doesn't throttle us.
    time.sleep(5)

def process_markdown_files(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,
                           hlsl_to_spirv_op, spirv_op_to_docs):
    rel_base_path = "win32/desktop-src/"

    md_files = md_table_to_dict(full_md_filepath)
           
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
                hl_op.name in hidden_intrinsics or hl_op.name in succesfully_created_issues):
            continue
        if hl_op.name not in md_files:
            emptyBody = ''
            create_github_issue_helper( hl_op.name, emptyBody, hlsl_to_dxil_op, dxil_op_to_docs, 
                                   hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,hlsl_to_spirv_op, spirv_op_to_docs)
            continue
        md_file = md_files[hl_op.name]
        md_file_parent_dir = 'direct3dhlsl' 
        if 'direct3d12' in md_file:
            md_file_parent_dir = 'direct3d12'
        
        filepath = os.path.join(pathlib.Path().resolve(),
                                rel_base_path,
                                md_file_parent_dir,
                                md_file[2])

        if filepath.endswith(".md"):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().split('# ', 1)[-1].split('\n',1)
                intrinsic_name = filter_escapes(content[0].split(' ', 1)[0].strip())  # First line as title
                body = content[1].strip() if len(content) > 1 else ''  # Rest as body
                body = replace_md_links_with_hyperlinks(body, md_file_parent_dir)
                if intrinsic_name == 'InstanceId':
                    intrinsic_name = 'InstanceID'
                create_github_issue_helper(intrinsic_name, body, hlsl_to_dxil_op, dxil_op_to_docs, 
                                           hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,
                                           hlsl_to_spirv_op, spirv_op_to_docs)
    
    save_issue_numbers(succesfully_created_issues)


def load_test_cases(intrinsic_subset= [], prefix=''):
    test_cases = {}
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
                hl_op.name in hidden_intrinsics):
            continue
        if prefix != '' and  hl_op.name not in intrinsic_subset:
            continue
        test_file_path = os.path.join(scratchpad_path, hl_op.name +f"{prefix}_test.hlsl")
        if os.path.exists(test_file_path):
            with open(test_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                test_cases[hl_op.name] = [content]
        count = 1
        test_file_path = os.path.join(scratchpad_path, hl_op.name + f"_{count}_test.hlsl")
        # hack because there are holes sometimes
        while(count < 35):
            if os.path.exists(test_file_path):
                with open(test_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    test_cases[hl_op.name].append(content)
            count = count + 1
            test_file_path = os.path.join(scratchpad_path, hl_op.name + f"_{count}_test.hlsl")

    return test_cases

def load_spirv_test_cases():
    return load_test_cases(special_spirv_example_code_intrinsics, '_spirv')

def load_all_test_cases():
    base_test_cases = load_test_cases()
    spirv_test_cases = load_spirv_test_cases()
    for hl_op_name, test_cases in spirv_test_cases.items():
        if hl_op_name in base_test_cases:
            base_test_cases[hl_op_name].append('### SPIRV Example(s):')
            base_test_cases[hl_op_name].extend(test_cases)
    return base_test_cases

def gen_test_case_body(intrinsic_name, hlsl_intrinsics_test_cases):
    body = ''
    if intrinsic_name in hlsl_intrinsics_test_cases:
        count = 1
        for test_case in hlsl_intrinsics_test_cases[intrinsic_name]:
            if test_case == '### SPIRV Example(s):':
                body = f'{body}\n {test_case}\n'
                continue
            body = f'{body}\n ### Example {count}\n```hlsl\n{test_case}\n```'
            count = count + 1
    return body

def create_group_github_issue(succesfully_created_issues):
    if(len(succesfully_created_issues) < 155):
        return
    title = 'Implement the entire HLSL API set.'
    if title not in succesfully_created_issues:
        labels = ['HLSL', 'backend:SPIR-V', 'backend:DirectX', 'metabug', 'bot:HLSL']
        body = ''
        for _, issue_id in succesfully_created_issues.items():
            body += f'- [ ] #{issue_id}\n'
        issue_number = create_github_issue(title, body, labels)
        if issue_number:
            succesfully_created_issues[title] = issue_number

if __name__ == "__main__":
    keys = ApiKeys.parse_api_keys()
    GITHUB_TOKEN = keys.github_key
    hlsl_ignore_intrinsics = []
    deprecated_intrinsics = gen_deprecated_intrinsics()
    hlsl_to_spirv_op = load_dict("hlsl_spirv_intrinsics.pkl", gen_spirv_shader_instr)
    spirv_op_to_docs = load_dict("hlsl_spirv_docs.pkl", parse_spirv_spec)
    hlsl_to_dxil_op = load_dict("hlsl_intrinsics.pkl", run_dxc)
    hlsl_intrinsics_test_cases = load_dict("hlsl_intrinsics_tests.pkl", load_all_test_cases)
    hlsl_completed_intrinsics = llvm_hlsl_completed_intrinsics()
    dxil_completed_ops = llvm_dxil_completed_ops()
    dxil_op_to_docs = query_dxil()
    hlsl_ignore_intrinsics = ['errorf', 'noise']
    hlsl_ignore_intrinsics.extend(deprecated_intrinsics)
    hlsl_ignore_intrinsics.extend(hlsl_completed_intrinsics)
    
    #temporary
    hlsl_ignore_intrinsics.extend(successfully_created_intrinsics)

    succesfully_created_issues = load_dict("issue_numbers.pkl")
    print(succesfully_created_issues)
    hlsl_ignore_intrinsics.extend(list(succesfully_created_issues.keys()))
    
    spirv_op_to_docs['NonSemantic.DebugPrintf'] = 'See "Using Debug Printf in HLSL Shaders" in https://vulkan.lunarg.com/doc/sdk/1.3.283.0/windows/debug_printf.html'

    try:
        process_markdown_files(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases, hlsl_to_spirv_op, spirv_op_to_docs)
    except KeyError as e:
        print(traceback.format_exc())
        print(f"KeyError: {e}")
        save_issue_numbers(succesfully_created_issues)
        exit(0)
    except IndexError as e:
        print(f"IndexError: {e}")
        save_issue_numbers(succesfully_created_issues)
        exit(0)
    
    create_group_github_issue(succesfully_created_issues)
