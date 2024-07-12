import json
import pickle
import os
import pathlib
from query import load_dict

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_json(file_path, jdata):
    with open(file_path, 'w') as file:
        json.dump(jdata, file, indent=4)


def save_dict_as_pickle(dict_data, rel_file_path):
    if len(dict_data) == 0:
        return
    file_path = os.path.join(pathlib.Path().resolve(), rel_file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(dict_data, f)

if __name__ == "__main__":
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    pickle_path = os.path.join(scratchpad_path, 'issue_numbers.pkl')
    json_path = os.path.join(scratchpad_path, 'issue_numbers.json')
    if not os.path.exists(pickle_path):
        os.makedirs(scratchpad_path, exist_ok=True)
        jdata = load_json(json_path)
        save_dict_as_pickle(jdata, 'scratch/issue_numbers.pkl')
    else:
        jdata = load_dict("issue_numbers.pkl")
        save_json(json_path, jdata)
        print(jdata)