import os
import re
from collections import defaultdict
import chardet
import json
from query import db_hlsl

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Function to search for function names in a file and return their line numbers
def search_functions_in_file(file_path, function_names):
    function_pattern = re.compile(r'\b(' + '|'.join(function_names) + r')\b')
    results = defaultdict(list)
    intrinsics_found = set()
    en = detect_encoding(file_path)
    with open(file_path, 'r+', encoding=en) as file:
        for line_number, line in enumerate(file, start=1):
            matches = function_pattern.findall(line)
            for match in matches:
                results[match].append(line_number)
                intrinsics_found.add(match)
    return (intrinsics_found, results)

# Function to recursively search through directory for files and search functions within them
def search_functions_in_directory(directory, function_names, extensions):
    file_function_map = defaultdict(dict)
    intrinsics_found = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                print(file_path)
                intrinsics_set, functions_found = search_functions_in_file(file_path, function_names)
                intrinsics_found = intrinsics_found.union(intrinsics_set)
                if functions_found:
                    file_function_map[file] = functions_found
                
    return intrinsics_found, file_function_map

# Define the directory, function names, and file extensions to search
directory = '/mnt/devDrive/Projects/DirectML'
function_names = []  # Add your list of function names here
extensions = ['.hlsl', '.hlsli', '.fx', '.fxh']

def gen_hlsl_intrinsic_names():
    for hl_op in db_hlsl.intrinsics:
        if hl_op.ns != "Intrinsics":
            continue
        function_names.append(hl_op.name)

gen_hlsl_intrinsic_names()
# Search the directory for the functions and print the results
intrinsics_found, results = search_functions_in_directory(directory, function_names, extensions)
print(json.dumps(results, indent=4))
print(intrinsics_found)

