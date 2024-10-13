import os
import git
import pathlib
from datetime import datetime, timedelta
import re
import sys
from collections import defaultdict

import plotly.graph_objects as go
import chardet

# pylint: disable=wrong-import-position
sys.path.append(os.path.join(os.getcwd(), "DirectXShaderCompiler/utils/hct"))
from DirectXShaderCompiler.utils.hct.hctdb_instrhelp import get_db_hlsl
from DirectXShaderCompiler.utils.hct.hctdb_instrhelp import get_db_dxil
# pylint: enable=wrong-import-position

db_dxil = get_db_dxil()
db_hlsl = get_db_hlsl()

is_cli = False

def print_cli(*args, **kwargs):
    global is_cli
    if is_cli:
        print( " ".join(map(str,args)), **kwargs)


def get_dict_of_dxil_ops():
    opcode_to_dxil = {}
    for dxil_inst in db_dxil.instr:
        if dxil_inst.dxil_op:
            opcode_to_dxil[dxil_inst.dxil_opid] = dxil_inst
    return opcode_to_dxil

def get_list_of_hlsl_intrinsics():
    hlsl_intrinsic_names = set()
    for hl_op in db_hlsl.intrinsics:
        if hl_op.ns != "Intrinsics":
            continue
        hlsl_intrinsic_names.add(hl_op.name)
    return hlsl_intrinsic_names

all_dxil_ops = get_dict_of_dxil_ops()
all_hlsl_intrinsic = get_list_of_hlsl_intrinsics()

def getopNames(opcodes):
    opcodeNames = []
    for opcode in opcodes:
        op = int(opcode)
        if op in all_dxil_ops:
            opcodeNames.append(all_dxil_ops[op].dxil_op)
        else:
            print(f"warning dxilop: {op} was not found in all_dxil_ops of size {len(all_dxil_ops)}")
    return opcodeNames

def extract_all_opcodes(text):
    matches = re.findall(r'def\s+\w+\s*:\s*DXILOpMapping<\s*(\d+)', text)
    if matches:
       return matches
    return []

def find_functions(function_name, text):
        # Construct the regular expression pattern dynamically
        pattern = r'\b{}\s*\(\s*([^)]*)\s*\);'.format(re.escape(function_name))
        match = re.search(pattern, text)
        return match

def extract_all_hlsl_intrinsics(text):
    hlsl_intrinsics = all_hlsl_intrinsic
    found_intrinsics = []
    for intrinsic in hlsl_intrinsics:
        # note: need a better regex here
        match = find_functions(intrinsic, text)
        if match:
            found_intrinsics.append(intrinsic)
    return found_intrinsics

def llvm_graph_hlsl_intrinsics(data=None):
    if not data:
        data =  llvm_git_hlsl_parser()
    dates = [date for date, _, _, _, _ in data]
    intrinsics = [ int_count for _, int_count, _, _, _ in data]
    percentages = [ percentage for _, _, percentage, _, _ in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=intrinsics, mode='lines', name='HLSL intrinsics', line_color='rgb(0,255,0)'))
    fig.add_trace(go.Scatter(x=dates, y=percentages, mode='lines', name='HLSL int %', line_color='rgb(0,255, 125)'))

    fig.update_layout(title='HLSL Intrinsics Over Time',
                   title_font=dict(color='green'),
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   legend_title_font_color='white',
                   font_color="white",
                   xaxis_color='white',
                   yaxis_color='white',
                   xaxis_title='Date',
                   yaxis_title='HLSL Intrinsics',
                   showlegend=True)

    if is_cli:
        fig.show()
    else:
        fig.write_image("scratch/hlsl_fig.png")
    return fig

def llvm_git_hlsl_parser():
    llvm_path = os.path.join(pathlib.Path().resolve(), 'llvm-project')
    hlsl_intrinsic_rel_path = 'clang/lib/Headers/hlsl/hlsl_intrinsics.h'
    
    hlsl_intrinsic_files = get_git_commits(llvm_path, hlsl_intrinsic_rel_path)
  
    data = [
    ['Date',  '# of Intrinsics', f"% of Intrinsics", "Intrinsics", "commit id"]
    ]
    intrinsics = []
    num_intrinsics = 0
    for date, file_version, git_sha in hlsl_intrinsic_files:
        intrinsics = extract_all_hlsl_intrinsics(file_version)
        num_intrinsics = len(intrinsics)
        if num_intrinsics > 0:
            row = [date.strftime('%Y-%m-%d'), num_intrinsics, round(100 * (num_intrinsics / len(all_hlsl_intrinsic)),2), ', '.join(intrinsics), git_sha]
            data.append(row)
            print_cli(f"Date: {row[0]} intrinsic count: {num_intrinsics}")
    
    print_cli(f"intrinsic completion % {data[-1][2]}")
    return data


def llvm_hlsl_completed_intrinsics():
    intrinsics = set()
    llvm_path = os.path.join(pathlib.Path().resolve(), 'llvm-project')
    hlsl_intrinsic_rel_path = 'clang/lib/Headers/hlsl/hlsl_intrinsics.h'
    hlsl_intrinsic_path = os.path.join(llvm_path, hlsl_intrinsic_rel_path)
    with open(hlsl_intrinsic_path, 'r', encoding='utf-8') as file:
        content = file.read()
        intrinsics = set(extract_all_hlsl_intrinsics(content))
    
    return intrinsics

def llvm_dxil_completed_ops():
    dxil_ops = set()
    llvm_path = os.path.join(pathlib.Path().resolve(), 'llvm-project')
    dxil_opcode_rel_path = 'llvm/lib/Target/DirectX/DXIL.td'
    dxil_ops_path = os.path.join(llvm_path, dxil_opcode_rel_path)
    with open(dxil_ops_path, 'r', encoding='utf-8') as file:
        content = file.read()
        dxil_ops = set(extract_all_opcodes(content))
    
    return dxil_ops

def llvm_git_dxil_parser():
    llvm_path = os.path.join(pathlib.Path().resolve(), 'llvm-project')
    dxil_opcode_rel_path = 'llvm/lib/Target/DirectX/DXIL.td'
    
    dxil_opcode_files = get_git_commits(llvm_path, dxil_opcode_rel_path)

    data = [
    ['Date',  '# of DXILOps', f"% of DXILOps", "DXILOp ids", "DXILOp Names", "commit id"]
    ]
    opcodes = []
    num_opcodes = 0
    for date, file_version, git_sha in dxil_opcode_files:
        opcodes = extract_all_opcodes(file_version)
        num_opcodes = len(opcodes)
        if num_opcodes > 0:
            row = [date.strftime('%Y-%m-%d'), num_opcodes, round(100 * (num_opcodes / len(all_dxil_ops)),2), ', '.join(opcodes), ', '.join(getopNames(opcodes)), git_sha]
            data.append(row)
            print_cli(f"Date: {row[0]} opcode count: {num_opcodes}")
    
    print_cli(f"opcode completion % {data[-1][2]}")
    return data

def llvm_graph_dxil_intrinsics(data=None):
    if not data:
        data =  llvm_git_dxil_parser()
    dates = [date for date, _, _, _, _, _ in data]
    dxilops = [ dxil_count for _, dxil_count, _, _, _, _ in data]
    percentages = [  int_per for _, _,  int_per, _, _, _ in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dxilops, mode='lines', name='DXIL Ops', line_color='rgb(0,255,0)'))
    fig.add_trace(go.Scatter(x=dates, y=percentages, mode='lines', name='DXIL Ops %', line_color='rgb(0,255,125)'))
    
    fig.update_layout(title='DXIL Opcodes Over Time',
                   title_font=dict(color='green'),
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   legend_title_font_color='white',
                   font_color="white",
                   xaxis_color='white',
                   yaxis_color='white',
                   xaxis_title='Date',
                   yaxis_title='# of Completed DXIL Opcodes',
                   showlegend=True)

    if is_cli:
        fig.show()
    else:
        fig.write_image("scratch/dxil_fig.png")
    return fig

def enforce_date_range(committed_datetime, current_date):
    if current_date >= committed_datetime:
        return abs(committed_datetime - current_date)
    return  timedelta.max

def get_git_commits(repo_path: str, file_path : str):
    repo = git.Repo(repo_path)
    commits = list(repo.iter_commits(paths=file_path))
    commits.sort(key=lambda x: x.committed_datetime)
    
    # define range
    first_commit_date = commits[0].committed_datetime
    last_commit_date = commits[-1].committed_datetime

    # Define a timedelta for a week
    week_delta = timedelta(weeks=1)


    file_versions = []
    current_date = first_commit_date
    while current_date <= (last_commit_date + week_delta):
        closest_commit = min(commits, key=lambda x: enforce_date_range(x.committed_datetime, current_date))
    
        # Get the version of the file in the closest commit

        raw_data = closest_commit.tree[file_path].data_stream.read()
        encoding = chardet.detect(raw_data)['encoding']
        version = raw_data.decode(encoding)
        file_versions.append((current_date, version, closest_commit.hexsha))
    
        current_date += week_delta
    return file_versions



def get_creation_date(repo, file_path):
    # Get the first commit for the file
    commits = list(repo.iter_commits(paths=file_path))
    if commits:
        creation_date = datetime.fromtimestamp(commits[-1].committed_date).strftime('%Y-%m-%d %H:%M:%S')
        return (creation_date, commits[-1].hexsha)
    else:
        return None

def get_file_creation_dates(repo_path, rel_folder_path):
    # Initialize the repository
    repo = git.Repo(repo_path)
    folder_path = os.path.join(repo_path, rel_folder_path)
    assert not repo.bare

    # Dictionary to store creation dates
    creation_dates = {}

    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            #file_path = os.path.join(root, file)
            #relative_path = os.path.relpath(file_path, folder_path)
            rel_path =os.path.join(rel_folder_path, file)
            creation_date = get_creation_date(repo, rel_path)
            dict_key = file.split('.ll')[0]
            if creation_date:
                # Convert the timestamp to a readable date format
                creation_dates[dict_key] = creation_date
            else:
                creation_dates[dict_key] = 'No commits found'

    return creation_dates

# Example usage
def get_spirv_intrinsics_created_dates():
    llvm_path = os.path.join(pathlib.Path().resolve(), 'llvm-project')
    rel_folder_path = 'llvm/test/CodeGen/SPIRV/hlsl-intrinsics/'
    creation_dates = get_file_creation_dates(llvm_path, rel_folder_path)
    
    return creation_dates

def get_spirv_opNames(hlsl_intrinsic_list):
    return hlsl_intrinsic_list

def llvm_git_spirv_parser():
    
    spirv_opcode_files = get_spirv_intrinsics_created_dates()

    data = [
    ['Date',  '# of SPIRV Ops', f"% of SPIRV Ops", "HLSL Instrinsic Names", "SPIRVOp Names", "commit id"]
    ]
    
    date_key_list = sorted(
    [(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"), (key, hash)) for key, (date_str, hash) in spirv_opcode_files.items()]
    )
    result = defaultdict(list)
    seen_keys = set()
    
    for date, key in date_key_list:
        seen_keys.add(key[0])
        result[date.strftime("%Y-%m-%d %H:%M:%S")] = (list(seen_keys), key[1])

    # Note we need a way to check spirv ops, this is a temporary substitute
    spirv_ops = all_dxil_ops
    result = dict(result)
    for date, (hlsl_intrinsics, git_sha) in result.items():
        num_intrinsics = len(hlsl_intrinsics)
        row = [date, num_intrinsics, round(100 * (num_intrinsics / len(spirv_ops)),2), ', '.join(hlsl_intrinsics), ', '.join(get_spirv_opNames(hlsl_intrinsics)), git_sha]
        data.append(row)
        print_cli(f"Date: {row[0]} opcode count: {num_intrinsics}")
 
    print_cli(f"opcode completion % {data[-1][2]}")
    return data


def llvm_graph_spirv_intrinsics(data=None):
    if not data:
        data =  llvm_git_spirv_parser()
    dates = [date for date, _, _, _, _, _ in data]
    dxilops = [ dxil_count for _, dxil_count, _, _, _, _ in data]
    percentages = [  int_per for _, _,  int_per, _, _, _ in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dxilops, mode='lines', name='SPIRV Ops', line_color='rgb(0,255,0)'))
    fig.add_trace(go.Scatter(x=dates, y=percentages, mode='lines', name='SPIRV Ops %', line_color='rgb(0,255,125)'))

    fig.update_layout(title='SPIRV Opcodes Over Time',
                   title_font=dict(color='green'),
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   legend_title_font_color='white',
                   font_color="white",
                   xaxis_color='white',
                   yaxis_color='white',
                   xaxis_title='Date',
                   yaxis_title='# of Completed SPIRV Opcodes',
                   showlegend=True)

    if is_cli:
        fig.show()
    else:
        fig.write_image("scratch/spirv_fig.png")
    return fig
