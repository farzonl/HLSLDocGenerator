"""
A utility file
"""

import json
import os
from dataclasses import dataclass
from typing import List

@dataclass
class ApiKeys:
    """A class to Track API Keys."""
    def __init__(self, open_ai_key, github_key):
        self.open_ai_key = open_ai_key
        self.github_key = github_key

    @staticmethod
    def parse_api_keys(api_key_path : str = 'api_keys.json') -> 'ApiKeys':
        """A function that parse & sets api keys off a json file or  enviornment argument."""
        keys = None
        try:
            api_keys_dict = read_from_json(api_key_path)
            keys = ApiKeys(api_keys_dict.get("open_ai_key",''), api_keys_dict.get("github_key",''))
        except FileNotFoundError:
            print("api_keys.json not found. reading from env args instead")
            keys = ApiKeys(os.getenv("open_ai_key"), os.getenv("github_key"))
        return keys

def read_from_json(filename: str) -> ApiKeys:
    """A function loads a json file."""
    with open(filename, 'r', encoding="utf-8") as json_file:
        return json.load(json_file)
