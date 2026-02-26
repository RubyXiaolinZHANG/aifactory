import torch


def rgb_to_hsv(rgb, eps=1e-8):
    """
    Differentiable RGB → HSV conversion.
    Assumes input range [0, 1] and outputs H/S/V in [0, 1].
    Fully tensor-based, no in-place operations, gradients flow completely.
    """
    # Clamp to avoid numerical issues from extreme values
    rgb = torch.clamp(rgb, 0, 1)
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val + eps

    # ---------- Hue (fully torch.where, no in-place assignment) ----------
    # Case 1: max == r
    mask_r = (max_val == r) & (diff > eps)
    h_r = ((g - b) / diff) % 6
    h_r = torch.where(mask_r, h_r, torch.zeros_like(h_r))

    # Case 2: max == g
    mask_g = (max_val == g) & (diff > eps)
    h_g = ((b - r) / diff) + 2
    h_g = torch.where(mask_g, h_g, torch.zeros_like(h_g))

    # Case 3: max == b
    mask_b = (max_val == b) & (diff > eps)
    h_b = ((r - g) / diff) + 4
    h_b = torch.where(mask_b, h_b, torch.zeros_like(h_b))

    # Combine all cases (mutually exclusive, so simple addition works)
    h = h_r + h_g + h_b
    h = h / 6.0           # Normalize to [0, 1]
    h = torch.clamp(h, 0, 1)

    # ---------- Saturation ----------
    s = torch.where(max_val > eps, diff / (max_val + eps), torch.zeros_like(max_val))
    s = torch.clamp(s, 0, 1)

    # ---------- Value ----------
    v = max_val
    v = torch.clamp(v, 0, 1)

    hsv = torch.stack([h, s, v], dim=1)
    return hsv


class HSVLoss(torch.nn.Module):
    """
    HSV color space constraint loss.
    Supports constraining only H/S channels, or all three, with per-channel weights.
    """
    def __init__(self, h_weight=1.0, s_weight=1.0, v_weight=0.0, loss=None):
        """
        Args:
            h_weight (float): weight for hue channel loss
            s_weight (float): weight for saturation channel loss
            v_weight (float): weight for value channel loss (usually not constrained)
            loss_type (str): 'l1' or 'l2'
        """
        super().__init__()
        self.h_weight = h_weight
        self.s_weight = s_weight
        self.v_weight = v_weight
        if loss is None:
            self._loss = torch.nn.L1Loss()
        else:
            self._loss = eval(loss)()


    def forward(self, pred_rgb, gt_rgb):
        """
        Args:
            pred_rgb (Tensor): predicted image, [B, 3, H, W], range [0, 1]
            gt_rgb   (Tensor): ground truth image, [B, 3, H, W], range [0, 1]
        Returns:
            loss (Tensor): scalar loss value
        """
        # RGB -> HSV
        pred_hsv = rgb_to_hsv(pred_rgb)
        gt_hsv   = rgb_to_hsv(gt_rgb)

        # Split channels
        pred_h, pred_s, pred_v = pred_hsv[:, 0, :, :], pred_hsv[:, 1, :, :], pred_hsv[:, 2, :, :]
        gt_h,   gt_s,   gt_v   = gt_hsv[:, 0, :, :],   gt_hsv[:, 1, :, :],   gt_hsv[:, 2, :, :]

        # Choose distance metric based on loss_type
        loss_h = self._loss(pred_h, gt_h)  # torch.nn.functional.l1_loss(pred_h, gt_h)
        loss_s = self._loss(pred_s, gt_s)  #  torch.nn.functional.l1_loss(pred_s, gt_s)
        loss_v = self._loss(pred_v, gt_v)  #  torch.nn.functional.l1_loss(pred_v, gt_v)

        # Weighted sum
        total_loss = (self.h_weight * loss_h +
                      self.s_weight * loss_s +
                      self.v_weight * loss_v)
        return total_loss


# ------------------- Usage example -------------------
if __name__ == '__main__':
    import cv2
    import numpy as np

    # Assume pred and gt are network outputs, normalized to [0, 1]
    image_file = r"D:\Program\ToGit\xiaomi\aifactory\tests\images\vimeo_00001_0003_im1.png"
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1) / 255.0
    noise_image = np.clip((image + np.random.normal(0, 50/255, image.shape)).round(), 0.0, 1.0)
    gt = torch.from_numpy(image).unsqueeze(dim=0)
    pred = torch.from_numpy(noise_image).unsqueeze(dim=0)
    pred.requires_grad_(True)
    # Instantiate loss function: constrain only hue and saturation, use L1 distance
    hsv_loss_fn = HSVLoss(h_weight=0.5, s_weight=0.5, v_weight=0.0, loss_type='l1')

    loss = hsv_loss_fn(pred, gt)
    print(f'HSV Loss: {loss.item()}')

    # Backpropagation example
    loss.backward()
