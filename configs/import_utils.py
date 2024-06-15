import os
import sys
import torch
from importlib import import_module
from easydict import EasyDict

def parse_config(config_path):
    assert os.path.isfile(config_path), "config_path is not a file"
    config_dir = os.path.dirname(config_path)
    config_file = os.path.basename(config_path)
    config_module_name, extension = os.path.splitext(config_file)
    assert extension == ".py", "File specified by config_path is not a Python file"
    sys.path.insert(0, config_dir)
    module = import_module(config_module_name)
    sys.path.pop(0)
    config = EasyDict()
    for key, value in module.__dict__.items():
        if key.startswith("__"):
            continue
        config[key] = value
    del sys.modules[config_module_name]
    return config