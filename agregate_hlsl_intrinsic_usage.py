import os
import sys
import re
from collections import defaultdict
import chardet
import json
from query import db_hlsl

intrinsic_count_map = {}
total_intrinsics = 0.0

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Function to search for function names in a file and return their line numbers
def search_functions_in_file(file_path, function_names):
    global total_intrinsics
    function_pattern = re.compile(r'\b(' + '|'.join(function_names) + r')\s*\(')
    results = defaultdict(list)
    intrinsics_found = set()
    en = detect_encoding(file_path)
    with open(file_path, 'r+', encoding=en) as file:
        for line_number, line in enumerate(file, start=1):
            matches = function_pattern.findall(line)
            for match in matches:
                results[match].append(line_number)
                intrinsics_found.add(match)
                if match in intrinsic_count_map:
                    intrinsic_count_map[match] = intrinsic_count_map[match]  + 1
                else:
                    intrinsic_count_map[match] = 1
                total_intrinsics = total_intrinsics + 1
    return (intrinsics_found, results)

# Function to recursively search through directory for files and search functions within them
def search_functions_in_directory(directory_path, function_names, extensions):
    file_function_map = defaultdict(dict)
    intrinsics_found = set()
    for root, _, files in os.walk(directory_path):
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
directory_path = ''
function_names = []  # Add your list of function names here
extensions = ['.hlsl', '.hlsli', '.fx', '.fxh']

def gen_hlsl_intrinsic_names():
    for hl_op in db_hlsl.intrinsics:
        if hl_op.ns != "Intrinsics":
            continue
        function_names.append(hl_op.name)

def check_directory_path():
    global directory_path
    # Check if the directory path argument is provided
    if len(sys.argv) < 2:
        print("Error: No directory path argument provided.")
        sys.exit(1)
    
    directory_path = sys.argv[1]

    # Check if the provided argument is a valid directory path
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory path.")
        sys.exit(1)
    
    print(f"Directory path '{directory_path}' is valid.")

if __name__ == "__main__":
    check_directory_path()
    gen_hlsl_intrinsic_names()
    # Search the directory for the functions and print the results
    intrinsics_found, results = search_functions_in_directory(directory_path, function_names, extensions)
    print("intrinsic usage per file per line number:")
    print(json.dumps(results, indent=4))
    print("Unique intrinsic usage:")
    print(intrinsics_found)
    sorted_intrinsic_count_map = dict(sorted({k: (v / total_intrinsics)*100.0 for k, v in intrinsic_count_map.items()}.items(), key=lambda item: item[1], reverse=True))
    print("percentage of intrinsic usage:")
    print(sorted_intrinsic_count_map)


