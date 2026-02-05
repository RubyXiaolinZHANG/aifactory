import torch
from .image_quanlity_metrics import PSNR, ChannelL1, PixelL1


METRICS = {"mse": torch.nn.MSELoss,
           "l1": torch.nn.L1Loss,
           "pixel_l1": PixelL1,
           "channel_l1": ChannelL1,
           "cross_entropy": torch.nn.CrossEntropyLoss,
           "psnr": PSNR}


__all__ = [METRICS]
