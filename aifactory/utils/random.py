import random


def split_dict_randomly(original_dict, ratio=0.5, seed=None):
    """
    Randomly split a dictionary into two dictionaries

    Parameters:
    original_dict: The original dictionary
    ratio: Distribution ratio for the first dictionary (default is 0.5)
    seed: Random seed for reproducible results

    Returns:
    (dict1, dict2): A tuple containing two new dictionaries
    """
    if seed is not None:
        random.seed(seed)

    keys = list(original_dict.keys())
    random.shuffle(keys)  # 随机打乱键的顺序

    # split point
    split_index = int(len(keys) * ratio)

    # create two separated dictionaries
    dict1_keys = keys[:split_index]
    dict2_keys = keys[split_index:]

    dict1 = {k: original_dict[k] for k in dict1_keys}
    dict2 = {k: original_dict[k] for k in dict2_keys}

    return dict1, dict2
