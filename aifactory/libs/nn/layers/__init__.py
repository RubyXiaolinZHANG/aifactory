import torch


ACTIVATIONS = {"relu": torch.nn.ReLU,
               "leaky_relu": torch.nn.LeakyReLU,
               "prelu": torch.nn.PReLU,
               "softmax": torch.nn.Softmax,
               "sigmoid": torch.nn.Sigmoid,
               "silu": torch.nn.Sigmoid}


__all__ = ["ACTIVATIONS"]