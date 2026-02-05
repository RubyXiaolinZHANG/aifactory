import os
import numpy as np
import cv2
from pathlib import Path
import ruamel.yaml


def save_image(file_name, bgr):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    cv2.imwrite(file_name, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_as_image(data, save_file):
    assert (data.ndim == 3 and data.shape[-1] == 3) or data.ndim == 2
    if data.dtype != np.uint8:
        data = np.round(data * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if Path(save_file).suffix == ".png":
        cv2.imwrite(save_file, data, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(save_file, data)

def save_dict2yaml(data, yaml_file, first_level_insert_space=False):
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)  # 设置缩进
    yaml.preserve_quotes = True
    if first_level_insert_space:
        config = ruamel.yaml.CommentedMap()
        for id, (key, val) in enumerate(data.items()):
            if not isinstance(val, (str, int, float)):
                config.yaml_set_comment_before_after_key(key, before='\n')
            config[key] = val
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)
    else:
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)