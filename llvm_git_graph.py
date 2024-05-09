import os
import git
import pathlib
from datetime import datetime, timedelta
import re
import sys

import plotly.graph_objects as go

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
    fig.add_trace(go.Scatter(x=dates, y=intrinsics, mode='lines', name='HLSL intrinsics'))
    fig.add_trace(go.Scatter(x=dates, y=percentages, mode='lines', name='HLSL int %'))

    fig.update_layout(title='HLSL Intrinsics Over Time',
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
    fig.add_trace(go.Scatter(x=dates, y=dxilops, mode='lines', name='DXIL Ops'))
    fig.add_trace(go.Scatter(x=dates, y=percentages, mode='lines', name='DXIL Ops %'))

    fig.update_layout(title='DXIL Opcodes Over Time',
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
        version = closest_commit.tree[file_path].data_stream.read().decode('utf-8')
        file_versions.append((current_date, version, closest_commit.hexsha))
    
        current_date += week_delta
    return file_versions

#llvm_git_hlsl_parser()