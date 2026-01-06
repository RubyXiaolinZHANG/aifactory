import os
from pathlib import Path
from ruamel.yaml import YAML
from pathlib import Path
import json
import numpy as np


def check_file(file):

    if isinstance(file, str):
        file = Path(file)
    elif isinstance(file, Path):
        pass
    else:
        assert ValueError("Do not support file path of tyep {}".format(type(file)))

    if not file.exists():
        raise FileNotFoundError(f"file does not exist: {file}")

    if not file.is_file():
        raise ValueError(f"the path is not a file: {file}")


def join_paths_safe(base, parts):
    """Safely join paths using pathlib"""
    base_path = Path(base)
    for part in parts:
        part_str = str(part).lstrip('/\\')
        base_path = base_path / part_str
    return str(base_path)


def join_path_constructor(loader, node):
    # Get the sequence values from the YAML node
    seq = loader.construct_sequence(node)
    # Join all items as a path
    return join_paths_safe(seq[0], seq[1:])
    # return os.path.join(*[str(item) for item in seq])


def load_file_yaml(file):
    check_file(file)
    with open(file, "r", encoding="utf-8") as file:
        yaml = YAML(typ='safe')
        yaml.allow_duplicate_keys = False
        yaml.Constructor.add_constructor('!join_path', join_path_constructor)
        data = yaml.load(file)
    return data


def load_file_json(file):
    check_file(file)
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_file_raw(file, dtype=np.uint16):
    check_file(file)
    return np.fromfile(file, dtype=dtype)


def load_file(file, suffix=None):
    suffix = Path(file).suffix if suffix is None else suffix
    if suffix == ".yaml":
        return load_file_yaml(file)
    elif suffix == ".json":
        return load_file_json(file)
    elif suffix == ".raw":
        return load_file_raw(file)
    else:
        raise ValueError("do not support loading file {}".format(suffix))

import torch

torch.nn.Upsample