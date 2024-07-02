import os
import requests
import pathlib
import pickle
import re
from query import run_dxc, db_hlsl, gen_deprecated_intrinsics, hidden_intrinsics, query_dxil, gen_spirv_shader_instr
from llvm_git_graph import llvm_hlsl_completed_intrinsics, llvm_dxil_completed_ops
from spirv_doc_fetch import parse_spirv_spec
from utils import ApiKeys
import time


# GitHub repository and personal access token
REPO_OWNER = 'farzonl'
REPO_NAME = 'test_repo'
GITHUB_TOKEN = ''

deprecated_intrinsics = []

# Directory containing markdown files
MD_FILES = [
   "direct3dhlsl/abort.md",
   "direct3dhlsl/dx-graphics-hlsl-abs.md",
   "direct3dhlsl/dx-graphics-hlsl-acos.md",
   "direct3dhlsl/dx-graphics-hlsl-all.md",
   "direct3dhlsl/allmemorybarrier.md",
   "direct3dhlsl/allmemorybarrierwithgroupsync.md",
   "direct3dhlsl/dx-graphics-hlsl-any.md",
   "direct3dhlsl/asdouble.md",
   "direct3dhlsl/dx-graphics-hlsl-asfloat.md",
   "direct3dhlsl/dx-graphics-hlsl-asin.md",
   "direct3dhlsl/dx-graphics-hlsl-asint.md",
   "direct3dhlsl/asuint.md",
   "direct3dhlsl/dx-graphics-hlsl-asuint.md",
   "direct3dhlsl/dx-graphics-hlsl-atan.md",
   "direct3dhlsl/dx-graphics-hlsl-atan2.md",
   "direct3dhlsl/dx-graphics-hlsl-ceil.md",
   "direct3dhlsl/dx-graphics-hlsl-clamp.md",
   "direct3dhlsl/dx-graphics-hlsl-clip.md",
   "direct3dhlsl/dx-graphics-hlsl-cos.md",
   "direct3dhlsl/dx-graphics-hlsl-cosh.md",
   "direct3dhlsl/countbits.md",
   "direct3dhlsl/dx-graphics-hlsl-cross.md",
   "direct3dhlsl/dx-graphics-hlsl-d3dcolortoubyte4.md",
   "direct3dhlsl/dx-graphics-hlsl-ddx.md",
   "direct3dhlsl/ddx-coarse.md",
   "direct3dhlsl/ddx-fine.md",
   "direct3dhlsl/dx-graphics-hlsl-ddy.md",
   "direct3dhlsl/ddy-coarse.md",
   "direct3dhlsl/ddy-fine.md",
   "direct3dhlsl/dx-graphics-hlsl-degrees.md",
   "direct3dhlsl/dx-graphics-hlsl-determinant.md",
   "direct3dhlsl/devicememorybarrier.md",
   "direct3dhlsl/devicememorybarrierwithgroupsync.md",
   "direct3dhlsl/dx-graphics-hlsl-distance.md",
   "direct3dhlsl/dx-graphics-hlsl-dot.md",
   "direct3dhlsl/dst.md",
   "direct3dhlsl/errorf.md",
   "direct3dhlsl/evaluateattributecentroid.md",
   "direct3dhlsl/evaluateattributeatsample.md",
   "direct3dhlsl/evaluateattributesnapped.md",
   "direct3dhlsl/dx-graphics-hlsl-exp.md",
   "direct3dhlsl/dx-graphics-hlsl-exp2.md",
   "direct3dhlsl/f16tof32.md",
   "direct3dhlsl/f32tof16.md",
   "direct3dhlsl/dx-graphics-hlsl-faceforward.md",
   "direct3dhlsl/firstbithigh.md",
   "direct3dhlsl/firstbitlow.md",
   "direct3dhlsl/dx-graphics-hlsl-floor.md",
   "direct3dhlsl/dx-graphics-hlsl-fma.md",
   "direct3dhlsl/dx-graphics-hlsl-fmod.md",
   "direct3dhlsl/dx-graphics-hlsl-frac.md",
   "direct3dhlsl/dx-graphics-hlsl-frexp.md",
   "direct3dhlsl/dx-graphics-hlsl-fwidth.md",
   "direct3dhlsl/dx-graphics-hlsl-getrendertargetsamplecount.md",
   "direct3dhlsl/dx-graphics-hlsl-getrendertargetsampleposition.md",
   "direct3dhlsl/groupmemorybarrier.md",
   "direct3dhlsl/groupmemorybarrierwithgroupsync.md",
   "direct3dhlsl/interlockedadd.md",
   "direct3dhlsl/interlockedand.md",
   "direct3dhlsl/interlockedcompareexchange.md",
   "direct3dhlsl/interlockedcomparestore.md",
   "direct3dhlsl/interlockedexchange.md",
   "direct3dhlsl/interlockedmax.md",
   "direct3dhlsl/interlockedmin.md",
   "direct3dhlsl/interlockedor.md",
   "direct3dhlsl/interlockedxor.md",
   "direct3dhlsl/dx-graphics-hlsl-isfinite.md",
   "direct3dhlsl/dx-graphics-hlsl-isinf.md",
   "direct3dhlsl/dx-graphics-hlsl-isnan.md",
   "direct3dhlsl/dx-graphics-hlsl-ldexp.md",
   "direct3dhlsl/dx-graphics-hlsl-length.md",
   "direct3dhlsl/dx-graphics-hlsl-lerp.md",
   "direct3dhlsl/dx-graphics-hlsl-lit.md",
   "direct3dhlsl/dx-graphics-hlsl-log.md",
   "direct3dhlsl/dx-graphics-hlsl-log10.md",
   "direct3dhlsl/dx-graphics-hlsl-log2.md",
   "direct3dhlsl/mad.md",
   "direct3dhlsl/dx-graphics-hlsl-max.md",
   "direct3dhlsl/dx-graphics-hlsl-min.md",
   "direct3dhlsl/dx-graphics-hlsl-modf.md",
   "direct3dhlsl/dx-graphics-hlsl-msad4.md",
   "direct3dhlsl/dx-graphics-hlsl-mul.md",
   "direct3dhlsl/dx-graphics-hlsl-noise.md",
   "direct3dhlsl/dx-graphics-hlsl-normalize.md",
   "direct3dhlsl/dx-graphics-hlsl-pow.md",
   "direct3dhlsl/printf.md",
   "direct3dhlsl/process2dquadtessfactorsavg.md",
   "direct3dhlsl/process2dquadtessfactorsmax.md",
   "direct3dhlsl/process2dquadtessfactorsmin.md",
   "direct3dhlsl/processisolinetessfactors.md",
   "direct3dhlsl/processquadtessfactorsavg.md",
   "direct3dhlsl/processquadtessfactorsmax.md",
   "direct3dhlsl/processquadtessfactorsmin.md",
   "direct3dhlsl/processtritessfactorsavg.md",
   "direct3dhlsl/processtritessfactorsmax.md",
   "direct3dhlsl/processtritessfactorsmin.md",
   "direct3dhlsl/dx-graphics-hlsl-radians.md",
   "direct3dhlsl/rcp.md",
   "direct3dhlsl/dx-graphics-hlsl-reflect.md",
   "direct3dhlsl/dx-graphics-hlsl-refract.md",
   "direct3dhlsl/reversebits.md",
   "direct3dhlsl/dx-graphics-hlsl-round.md",
   "direct3dhlsl/dx-graphics-hlsl-rsqrt.md",
   "direct3dhlsl/dx-graphics-hlsl-saturate.md",
   "direct3dhlsl/dx-graphics-hlsl-sign.md",
   "direct3dhlsl/dx-graphics-hlsl-sin.md",
   "direct3dhlsl/dx-graphics-hlsl-sincos.md",
   "direct3dhlsl/dx-graphics-hlsl-sinh.md",
   "direct3dhlsl/dx-graphics-hlsl-smoothstep.md",
   "direct3dhlsl/dx-graphics-hlsl-sqrt.md",
   "direct3dhlsl/dx-graphics-hlsl-step.md",
   "direct3dhlsl/dx-graphics-hlsl-tan.md",
   "direct3dhlsl/dx-graphics-hlsl-tanh.md",
   "direct3dhlsl/dx-graphics-hlsl-tex1d.md",
   "direct3dhlsl/dx-graphics-hlsl-tex1d-s-t-ddx-ddy.md",
   "direct3dhlsl/dx-graphics-hlsl-tex1dbias.md",
   "direct3dhlsl/dx-graphics-hlsl-tex1dgrad.md",
   "direct3dhlsl/dx-graphics-hlsl-tex1dlod.md",
   "direct3dhlsl/dx-graphics-hlsl-tex1dproj.md",
   "direct3dhlsl/dx-graphics-hlsl-tex2d.md",
   "direct3dhlsl/dx-graphics-hlsl-tex2d-s-t-ddx-ddy.md",
   "direct3dhlsl/dx-graphics-hlsl-tex2dbias.md",
   "direct3dhlsl/dx-graphics-hlsl-tex2dgrad.md",
   "direct3dhlsl/dx-graphics-hlsl-tex2dlod.md",
   "direct3dhlsl/dx-graphics-hlsl-tex2dproj.md",
   "direct3dhlsl/dx-graphics-hlsl-tex3d.md",
   "direct3dhlsl/dx-graphics-hlsl-tex3d-s-t-ddx-ddy.md",
   "direct3dhlsl/dx-graphics-hlsl-tex3dbias.md",
   "direct3dhlsl/dx-graphics-hlsl-tex3dgrad.md",
   "direct3dhlsl/dx-graphics-hlsl-tex3dlod.md",
   "direct3dhlsl/dx-graphics-hlsl-tex3dproj.md",
   "direct3dhlsl/dx-graphics-hlsl-texcube.md",
   "direct3dhlsl/dx-graphics-hlsl-texcube-s-t-ddx-ddy.md",
   "direct3dhlsl/dx-graphics-hlsl-texcubebias.md",
   "direct3dhlsl/dx-graphics-hlsl-texcubegrad.md",
   "direct3dhlsl/dx-graphics-hlsl-texcubelod.md",
   "direct3dhlsl/dx-graphics-hlsl-texcubeproj.md",
   "direct3dhlsl/dx-graphics-hlsl-transpose.md",
   "direct3dhlsl/dx-graphics-hlsl-trunc.md",
   "direct3dhlsl/wavegetlanecount.md",
   "direct3dhlsl/wavegetlaneindex.md",
   "direct3dhlsl/waveisfirstlane.md",
   "direct3dhlsl/waveanytrue.md",
   "direct3dhlsl/wavealltrue.md",
   "direct3dhlsl/waveballot.md",
   "direct3dhlsl/wavereadlaneat.md",
   "direct3dhlsl/wavereadfirstlane.md",
   "direct3dhlsl/waveactiveallequal.md",
   "direct3dhlsl/waveallbitand.md",
   "direct3dhlsl/waveallbitor.md",
   "direct3dhlsl/waveallbitxor.md",
   "direct3dhlsl/waveactivecountbits.md",
   "direct3dhlsl/waveallmax.md",
   "direct3dhlsl/waveallmin.md",
   "direct3dhlsl/waveallproduct.md",
   "direct3dhlsl/waveallsum.md",
   "direct3dhlsl/waveprefixcountbytes.md",
   "direct3dhlsl/waveprefixsum.md",
   "direct3dhlsl/waveprefixproduct.md",
   "direct3dhlsl/quadreadlaneat.md",
   "direct3dhlsl/quadreadacrossdiagonal.md",
   "direct3dhlsl/quadswapx.md",
   "direct3dhlsl/quadswapy.md",
   "direct3d12/traceray-function.md",
   "direct3d12/reporthit-function.md",
   "direct3d12/callshader-function.md",
   "direct3d12/ignorehit-function.md",
   "direct3d12/accepthitandendsearch-function.md",
   "direct3d12/dispatchraysindex.md",
   "direct3d12/dispatchraysdimensions.md",
   "direct3d12/worldrayorigin.md",
   "direct3d12/worldraydirection.md",
   "direct3d12/objectrayorigin.md",
   "direct3d12/objectraydirection.md",
   "direct3d12/raytmin.md",
   "direct3d12/raytcurrent.md",
   "direct3d12/primitiveindex.md",
   "direct3d12/instanceid.md",
   "direct3d12/instanceindex.md",
   "direct3d12/hitkind.md",
   "direct3d12/rayflags.md",
   "direct3d12/objecttoworld3x4.md",
   "direct3d12/worldtoobject3x4.md",
   "direct3d12/objecttoworld4x3.md",
   "direct3d12/worldtoobject4x3.md",
]

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
#"ReportHit",
#"CallShader",
#"IgnoreHit",
#"AcceptHitAndEndSearch",
#"DispatchRaysIndex",
#"DispatchRaysDimensions",
#"WorldRayOrigin",
#"WorldRayDirection",
#"ObjectRayOrigin",
#"ObjectRayDirection",
#"RayTMin",
#"RayTCurrent",
#"PrimitiveIndex",
#"InstanceIndex",
#"HitKind",
#"RayFlags",
#"ObjectToWorld3x4",
#"WorldToObject3x4",
#"ObjectToWorld4x3",
#"WorldToObject4x3"
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
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print(f"Successfully created issue: {title}")
    else:
        print(f"Failed to create issue: {title}")
        print(f"Response: {response.content}")

def create_github_issue_helper(intrinsic_name, body, hlsl_to_dxil_op, dxil_op_to_docs, 
                               hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,
                               hlsl_to_spirv_op, spirv_op_to_docs):
    if intrinsic_name in hlsl_ignore_intrinsics:
        return
    labels = ['HLSL']
    requirements_body  = f'- [ ] Implement `{intrinsic_name}` clang builtin,\n'
    requirements_body += f'- [ ] Link `{intrinsic_name}` clang builtin with `hlsl_intrinsics.h`\n'
    requirements_body += f'- [ ] Add sema checks for `{intrinsic_name}` to `CheckHLSLBuiltinFunctionCall` in `SemaChecking.cpp`\n'
    requirements_body += f'- [ ] Add codegen for `{intrinsic_name}` to `EmitHLSLBuiltinExpr` in `CGBuiltin.cpp`\n'
    requirements_body += f'- [ ] Add codegen tests to `clang/test/CodeGenHLSL/builtins/{intrinsic_name}.hlsl`\n'
    requirements_body += f'- [ ] Add sema tests to `clang/test/SemaHLSL/BuiltIns/{intrinsic_name}-errors.hlsl`\n'
    dxil_op =  hlsl_to_dxil_op[intrinsic_name]
    doc_body = ''
    if dxil_op != -1:
        dxil_docs = dxil_op_to_docs[dxil_op]
        requirements_body += f'- [ ] Create the `int_dx_{intrinsic_name}` intrinsic in `IntrinsicsDirectX.td`\n'
        requirements_body += f'- [ ] Create the `DXILOpMapping` of `int_dx_{intrinsic_name}` to  `{dxil_op}` in `DXIL.td`\n'
        requirements_body += f'- [ ] Create the  `{intrinsic_name}.ll` and `{intrinsic_name}_errors.ll` tests in `llvm/test/CodeGen/DirectX/`\n'

        dxil_doc_body =    '| DXIL Opcode | DXIL OpName | Shader Model | Shader Stages |'
        dxil_doc_body += '\n| ----------- | ----------- | ------------ | ------------- |'
        dxil_doc_body += f'\n| {dxil_op} | {dxil_docs[0]} | {dxil_docs[1]} | {dxil_docs[3]} |' 
        doc_body += '## Direct X\n\n' + dxil_doc_body + '\n'
        labels.append('backend:DirectX')
    
    spirv_op =  hlsl_to_spirv_op.get(intrinsic_name, -1)
    if spirv_op != -1:
        spirv_doc = spirv_op_to_docs[spirv_op]
        requirements_body += f'- [ ] Create the `int_spv_{intrinsic_name}` intrinsic in `IntrinsicsSPIRV.td`\n'
        requirements_body += f'- [ ] In SPIRVInstructionSelector.cpp create the {intrinsic_name} lowering and map  it to `int_spv_{intrinsic_name}` in ` SPIRVInstructionSelector::selectIntrinsic.\n'
        requirements_body += f'  [ ] Create SPIR-V backend test case in llvm/test/CodeGen/SPIRV/hlsl-intrinsics/{intrinsic_name}.ll\n'
        doc_body += '\n## SPIR-V\n\n' + spirv_doc + '\n'
        labels.append('backend:SPIR-V')
    
    test_case_body = gen_test_case_body(intrinsic_name, hlsl_intrinsics_test_cases)
    body = f'{requirements_body}\n{doc_body}\n{test_case_body}\n{body}' 
    title = f"Implement the `{intrinsic_name}` HLSL Function"
    
    create_github_issue(title, body, labels)
    # waiting so github doesn't throttle us.
    time.sleep(5)

def process_markdown_files(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,
                           hlsl_to_spirv_op, spirv_op_to_docs):
    rel_base_path = "win32/desktop-src/"
    doc_created_intrinsic = []
    for md_file in MD_FILES:
        filepath = os.path.join(pathlib.Path().resolve(),
                                rel_base_path,
                                md_file)
        md_file_parent_dir = 'direct3dhlsl' 
        if 'direct3d12' in filepath:
            md_file_parent_dir = 'direct3d12' 
        if filepath.endswith(".md"):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().split('# ', 1)[-1].split('\n',1)
                intrinsic_name = filter_escapes(content[0].split(' ', 1)[0].strip())  # First line as title
                body = content[1].strip() if len(content) > 1 else ''  # Rest as body
                body = replace_md_links_with_hyperlinks(body, md_file_parent_dir)
                doc_created_intrinsic.append(intrinsic_name)
                create_github_issue_helper(intrinsic_name, body, hlsl_to_dxil_op, dxil_op_to_docs, 
                                           hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases,
                                           hlsl_to_spirv_op, spirv_op_to_docs)

    emptyBody = ''           
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
                hl_op.name in hidden_intrinsics):
            continue
        if  hl_op.name in doc_created_intrinsic:
            continue
        create_github_issue_helper( hl_op.name, emptyBody, dxil_op_to_docs, hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases)


# Function to serialize the dictionary
def serialize_dict(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

# Function to deserialize the dictionary
def deserialize_dict(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_dict(file_name, runner):
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    file_path = os.path.join(scratchpad_path, file_name)
    ret_dict = {}
    if os.path.exists(file_path):
        ret_dict = deserialize_dict(file_path)
    else:
        ret_dict = runner()
        serialize_dict(ret_dict, file_path)
    return ret_dict

def load_test_cases():
    test_cases = {}
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    
    for hl_op in db_hlsl.intrinsics:
        if (hl_op.ns != "Intrinsics" or hl_op.name in deprecated_intrinsics or
                hl_op.name in hidden_intrinsics):
            continue
        test_file_path = os.path.join(scratchpad_path, hl_op.name + "_test.hlsl")
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

def gen_test_case_body(intrinsic_name, hlsl_intrinsics_test_cases):
    body = ''
    if intrinsic_name in hlsl_intrinsics_test_cases:
        count = 1
        for test_case in hlsl_intrinsics_test_cases[intrinsic_name]:
            body = f'{body}\n ### Example {count}\n```hlsl\n{test_case}\n```'
            count = count + 1
    return body


if __name__ == "__main__":
    keys = ApiKeys.parse_api_keys()
    GITHUB_TOKEN = keys.github_key
    hlsl_ignore_intrinsics = []
    deprecated_intrinsics = gen_deprecated_intrinsics()
    hlsl_to_dxil_op = load_dict("hlsl_intrinsics.pkl", run_dxc)
    hlsl_intrinsics_test_cases = load_dict("hlsl_intrinsics_tests.pkl", load_test_cases)
    hlsl_to_spirv_op = load_dict("hlsl_spirv_intrinsics.pkl", gen_spirv_shader_instr)
    spirv_op_to_docs = load_dict("hlsl_spirv_docs.pkl", parse_spirv_spec)
    hlsl_completed_intrinsics = llvm_hlsl_completed_intrinsics()
    dxil_completed_ops = llvm_dxil_completed_ops()
    dxil_op_to_docs = query_dxil()
    hlsl_ignore_intrinsics = ['errorf', 'noise', 'printf', 'InstanceId']
    hlsl_ignore_intrinsics.extend(deprecated_intrinsics)
    hlsl_ignore_intrinsics.extend(hlsl_completed_intrinsics)
    hlsl_ignore_intrinsics.extend(successfully_created_intrinsics)
    process_markdown_files(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_ignore_intrinsics, hlsl_intrinsics_test_cases, hlsl_to_spirv_op, spirv_op_to_docs)
