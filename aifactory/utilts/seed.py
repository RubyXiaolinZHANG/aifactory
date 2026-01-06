import torch
import numpy as np
import random
import os
import time


def set_seed(seed=42, deterministic=True):
    """
    Set all random seeds for reproducible results.

    Parameters:
        seed (int): Random seed, defaults to 42.
        deterministic (bool): Whether to use deterministic algorithms,
                              may impact performance but enhances reproducibility.
    """
    # 1. Python built-in random module
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Enable deterministic CUDA algorithms (may impact performance)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    # 5. Set PyTorch random number generator state
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)

    # 6. Operating system environment variable (some operations may depend on this)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seeds have been set to: {seed}")
    if deterministic:
        print("Deterministic mode enabled (may reduce performance but ensures reproducibility)")
    time.sleep(1)
    return torch_rng