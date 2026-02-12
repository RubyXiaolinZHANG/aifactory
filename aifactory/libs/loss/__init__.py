import torch
from .gradient_loss import GradientMagnitudePhaseLoss
from .masked_loss import MaskedL1Loss, MaskedMSELoss
from .color_loss import RawColorBindingLoss

LOSSES = {"mse": torch.nn.MSELoss,
          "l1": torch.nn.L1Loss,
          "pixel_l1": torch.nn.L1Loss,
          "smooth_l1": torch.nn.SmoothL1Loss,
          "cross_entropy": torch.nn.CrossEntropyLoss,
          "gradient": GradientMagnitudePhaseLoss,
          "masked_l1": MaskedL1Loss,
          "masked_mse": MaskedMSELoss,
          "color": RawColorBindingLoss}

__all__ = [LOSSES]
