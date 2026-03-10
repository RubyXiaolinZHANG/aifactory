import torch
from .dynamic_tanh import DynamicTanh
from .dynamic_erf import DynamicErf


ACTIVATIONS = {"relu": torch.nn.ReLU,
               "leaky_relu": torch.nn.LeakyReLU,
               "prelu": torch.nn.PReLU,
               "softmax": torch.nn.Softmax,
               "sigmoid": torch.nn.Sigmoid,
               "silu": torch.nn.SiLU,
               "gelu": torch.nn.GELU,
               "dyt": DynamicTanh,
               "dye": DynamicErf}


def get_activation_by_name(name, params=None):
    assert name in ACTIVATIONS
    if params is None:
        activation = ACTIVATIONS[name.lower()]()
    elif isinstance(params, (tuple, list)):
        activation = ACTIVATIONS[name.lower()](*params)
    elif isinstance(params, dict):
        activation = ACTIVATIONS[name.lower()](**params)
    else:
        raise ValueError("Do not support activation parameters type: {}".format(type(params)))
    return activation