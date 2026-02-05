import torch


def vst(x, shot, read):
    y = 2.0 / shot * torch.sqrt(torch.clip(shot * x + 3.0 / 8.0 * shot * shot + read, 0))
    return y


def ivst(y, shot, read):
    x= ((shot * y / 2.0) * (shot * y /2.0) - 3.0 / 8.0 * shot * shot - read) / shot
    return x
