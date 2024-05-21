import os
import requests
import pathlib
import pickle

from query import run_dxc, db_hlsl, deprecated_intrinsics, hidden_intrinsics, query_dxil
from llvm_git_graph import llvm_hlsl_completed_intrinsics, llvm_dxil_completed_ops
from utils import ApiKeys

# GitHub repository and personal access token
REPO_OWNER = 'farzonl'
REPO_NAME = 'test_repo'
GITHUB_TOKEN = ''

# Directory containing markdown files
MD_FILES = ['direct3dhlsl/dx-graphics-hlsl-acos.md',
            'direct3dhlsl/dx-graphics-hlsl-abs.md',
            "direct3dhlsl/dx-graphics-hlsl-mul.md"]

def create_github_issue(title, body):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print(f"Successfully created issue: {title}")
    else:
        print(f"Failed to create issue: {title}")
        print(f"Response: {response.content}")

def process_markdown_files(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_completed_intrinsics, hlsl_intrinsics_test_cases):
    rel_base_path = "win32/desktop-src/"
   
    #for filename in os.listdir(directory):
    for md_file in MD_FILES:
        filepath = os.path.join(pathlib.Path().resolve(),
                                rel_base_path,
                                md_file)
        if filepath.endswith(".md"):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().split('#', 1)[-1].split('\n',1)
                intrinsic_name = content[0].strip()  # First line as title
                if intrinsic_name in hlsl_completed_intrinsics:
                    continue

                dxil_op =  hlsl_to_dxil_op[intrinsic_name]
                dxil_docs = dxil_op_to_docs[dxil_op] if dxil_op != -1 else ["", "", "", "", ""]
                dxil_doc_body = ''
                if dxil_op != -1:
                    dxil_doc_body =    '| DXIL Opcode | DXIL OpName | Shader Model | Shader Stages |'
                    dxil_doc_body += '\n| ----------- | ----------- | ------------ | ------------- |'
                    dxil_doc_body +=  f'\n| {dxil_op} | {dxil_docs[0]} | {dxil_docs[1]} | {dxil_docs[3]} |' 
                test_case_body = gen_test_case_body(intrinsic_name, hlsl_intrinsics_test_cases)
                body = content[1].strip() if len(content) > 1 else ''  # Rest as body
                body = f'{dxil_doc_body}\n{test_case_body}\n{body}' 
                title = f"create {intrinsic_name} HLSL Intrinsic"

                create_github_issue(title, body)

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

    hlsl_to_dxil_op = load_dict("hlsl_intrinsics.pkl", run_dxc)
    hlsl_intrinsics_test_cases = load_dict("hlsl_intrinsics_tests.pkl", load_test_cases)
    hlsl_completed_intrinsics = llvm_hlsl_completed_intrinsics()
    dxil_completed_ops = llvm_dxil_completed_ops()
    dxil_op_to_docs = query_dxil()
    process_markdown_files(hlsl_to_dxil_op, dxil_op_to_docs, hlsl_completed_intrinsics, hlsl_intrinsics_test_cases)
