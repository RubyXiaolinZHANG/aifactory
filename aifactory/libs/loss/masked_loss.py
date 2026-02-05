import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """Compute MSE loss only in masked regions"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask):
        """
        Args:
            pred: Prediction tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            mask: Mask tensor [B, 1, H, W] or [B, H, W], where 1 indicates regions to compute loss
        """

        valid_pixels = (mask > 0).sum()  # Avoid division by zero
        if valid_pixels == 0:
            return torch.tensor(0, dtype=torch.float32, device=pred.device)
        # Ensure correct mask shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        # Expand to match pred's channel dimension
        if mask.shape[1] != pred.shape[1]:
            mask = mask.expand_as(pred)  # Copy to all channels

        # Compute squared error
        if pred.device != target.device:
            target = target.to(pred.device)
        squared_error = (pred - target) ** 2

        # Apply mask
        if pred.device != mask.device:
            mask = mask.to(pred.device)
        masked_error = squared_error * mask

        # Calculate loss
        if self.reduction == 'mean':
            # Average only over masked regions
            loss = masked_error.sum() / valid_pixels
        elif self.reduction == 'sum':
            loss = masked_error.sum()
        else:  # 'none'
            loss = masked_error

        return loss


class MaskedL1Loss(nn.Module):
    """Compute L1 loss only in masked regions"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask):
        # Expand to match pred's channel dimension
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Expand to match pred's channel dimension
        if mask.shape[1] != pred.shape[1]:
            mask = mask.expand_as(pred)

        # Compute abs error
        l1_error = torch.abs(pred - target)

        # Apply mask
        masked_error = l1_error * mask

        # Calculate loss
        if self.reduction == 'mean':
            valid_pixels = mask.sum() + 1e-8
            loss = masked_error.sum() / valid_pixels
        elif self.reduction == 'sum':
            loss = masked_error.sum()
        else:
            loss = masked_error

        return loss


def test_op():
    # test example
    pred = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)
    mask = torch.randint(0, 2, (4, 1, 32, 32)).float()  # 0/1掩码

    loss_mse = MaskedMSELoss()
    loss = loss_mse(pred, target, mask)
    print(f"Masked MSE Loss: {loss.item():.4f}")

    loss_l1 = MaskedL1Loss()
    loss = loss_l1(pred, target, mask)
    print(f"Masked L1 Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_op()