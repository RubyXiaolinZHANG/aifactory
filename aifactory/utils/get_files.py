import os
from pathlib import Path


def get_files_by_key(folder, key_in_file):
    pass


def get_files_by_extension(folder, suffix):
    if isinstance(suffix, str):
        suffix = [suffix]
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix in suffix:
                all_files.append(os.path.join(root, file).replace("\\", "/"))
    return all_files


def get_files_by_key_extension(folder, suffix):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    return all_files


def get_target_files(folder, key_in_filename=None, suffix=None):
    assert key_in_filename is not None or suffix is not None
    if key_in_filename is not None and suffix is not None:
        return get_files_by_key_extension(folder, key_in_filename, suffix)
    elif key_in_filename is not None:
        return get_files_by_key(folder, key_in_filename)
    elif suffix is not None:
        return get_files_by_extension(folder, suffix)
    else:
        pass
