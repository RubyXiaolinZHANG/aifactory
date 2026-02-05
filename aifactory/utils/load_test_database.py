import os
import re
from .get_files import get_target_files


def natural_sort_key(filename):
    """自然排序键函数：将数字部分转换为整数"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


def natural_sort_with_indices(filenames):
    """自然排序并返回索引"""
    # 使用自然排序键
    indexed = list(enumerate(filenames))
    sorted_indexed = sorted(indexed, key=lambda x: natural_sort_key(x[1]))

    indices = [idx for idx, _ in sorted_indexed]
    sorted_names = [name for _, name in sorted_indexed]

    return sorted_names, indices


def read_raw_from_folder(path, raw_suffix=".raw"):
    raw_files = get_target_files(path, suffix=raw_suffix)
    if raw_files is None or len(raw_files) < 1:
        return None
    samples = {}

    for raw_file in raw_files:
        raw_file = raw_file.replace("\\", "/")
        meta_file = raw_file.replace(raw_suffix, ".json")
        assert os.path.exists(meta_file)
        sample_name = raw_file.split("/")[-2]
        if sample_name in samples:
            samples[sample_name]["raw"].append(raw_file)
            samples[sample_name]["meta"].append(meta_file)
        else:
            samples[sample_name] = {"raw": [raw_file],
                                    "meta": [meta_file]}
    for sample_name, datas in samples.items():
        sorted_raws, indices = natural_sort_with_indices(datas["raw"])
        sorted_metas = [datas["meta"][i] for i in indices]
        samples[sample_name] = {i:{"raw": raw_file, "meta": meta_file}
                                for i, (raw_file, meta_file) in enumerate(zip(sorted_raws, sorted_metas))}
    if len(samples) < 1:
        samples = None
    return samples


def read_raw_from_folders(paths, raw_suffix=".raw"):
    if isinstance(paths, str):
        paths = [paths]
    samples = {}
    for path in paths:
        samples_ = read_raw_from_folder(path, raw_suffix)
        if samples_ is None:
            continue
        else:
            samples.update(samples_)
    if len(samples) == 0:
        return None
    else:
        return samples

def generate_control_scene_raw():
    pass


DATABASE_LOADERS = {"read_raw_from_folders": read_raw_from_folders,
                    "generate_control_scene_raw": generate_control_scene_raw}
