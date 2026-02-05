import torch
from .gradient_loss import GradientMagnitudePhaseLoss
from .masked_loss import MaskedL1Loss, MaskedMSELoss

LOSSES = {"mse": torch.nn.MSELoss,
          "l1": torch.nn.L1Loss,
          "pixel_l1": torch.nn.L1Loss,
          "smooth_l1": torch.nn.SmoothL1Loss,
          "cross_entropy": torch.nn.CrossEntropyLoss,
          "gradient": GradientMagnitudePhaseLoss,
          "masked_l1": MaskedL1Loss,
          "masked_mse": MaskedMSELoss}

__all__ = [LOSSES]
