import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientMagnitudePhaseLoss(nn.Module):
    """
    A loss function that combines gradient magnitude and phase (direction).
    Adopts a parallel constraint method, calculating magnitude loss and phase
    loss separately, then summing them with weights.
    """

    def __init__(self, magnitude_weight=1.0, phase_weight=0.5):
        """
        Args:
            magnitude_weight (float): Weight for the gradient magnitude loss.
            phase_weight (float): Weight for the gradient phase (direction) loss.
        """
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight

        # Using Sobel operator for gradient computation (more accurate)
        self.sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                            [-2., 0., 2.],
                                            [-1., 0., 1.]]).view(1, 1, 3, 3) / 4
        self.sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                            [0., 0., 0.],
                                            [1., 2., 1.]]).view(1, 1, 3, 3) / 4

    def get_gradient(self, img):
        """Computes the gradient magnitude and phase (direction) of an image."""
        n, c, h, w = img.shape
        # Ensure kernels are on the correct device and data type
        kernel_x = self.sobel_kernel_x.to(img.device).type_as(img).repeat(c, 1, 1, 1)
        kernel_y = self.sobel_kernel_y.to(img.device).type_as(img).repeat(c, 1, 1, 1)

        # Compute gradients in x and y directions
        grad_x = F.conv2d(img, kernel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, kernel_y, padding=1, groups=img.shape[1])

        # Compute gradient magnitude: sqrt(gx^2 + gy^2)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))

        # Compute gradient phase (direction): arctan(gy / gx), result in [-pi, pi]
        phase = torch.atan2(grad_y, grad_x)

        return magnitude, phase, grad_x, grad_y

    def forward(self, output, target):
        """
        Computes the combined gradient magnitude and phase loss.

        Args:
            output (torch.Tensor): Network output image (B, C, H, W).
            target (torch.Tensor): Target image (B, C, H, W).

        Returns:
            torch.Tensor: The loss value.
            dict: A dictionary containing individual loss components (for monitoring).
        """

        # 1. Compute gradient magnitude and phase for output and target separately
        mag_out, phase_out, _, _ = self.get_gradient(output)
        mag_target, phase_target, _, _ = self.get_gradient(target)

        # 2. Compute magnitude loss (using L1 loss, more robust to outliers)
        mag_loss = F.l1_loss(mag_out, mag_target)

        # 3. Compute phase (direction) loss
        # Note: Phase is an angle. Direct subtraction can cause issues due to 2π periodicity.
        # Solution: Compute the cosine difference of angles, converting to a loss in [0, 2]
        # cos(theta1 - theta2) = cos(theta1)*cos(theta2) + sin(theta1)*sin(theta2)
        # Loss = 1 - cos(theta1 - theta2). Loss is 0 when angle diff is 0, max 2 when diff is π.
        cos_angle_diff = torch.cos(phase_out - phase_target)
        # Clamp cos value to [-1, 1] to prevent floating-point errors
        cos_angle_diff = torch.clamp(cos_angle_diff, -1.0, 1.0)
        phase_loss = 1.0 - cos_angle_diff.mean()

        # 4. Weighted sum to obtain the total loss
        total_loss = (self.magnitude_weight * mag_loss) + (self.phase_weight * phase_loss)

        # Return total loss and its components (for monitoring)
        loss_dict = {
            'total_grad_loss': total_loss.item(),
            'magnitude_loss': mag_loss.item(),
            'phase_loss': phase_loss.item()
        }

        return total_loss, loss_dict


# Usage example
if __name__ == "__main__":
    import cv2
    from aifactory.libs.data.control_scenes.simulated_images import get_stripe_image

    # Simulated data
    height, width = 256, 256
    gray_1 = np.random.randint(0, 256, 3).astype(np.uint8)
    gray_2 = np.random.randint(0, 256, 3).astype(np.uint8)
    target = get_stripe_image(height, width, 15,gray_1=gray_1,gray_2=gray_2)
    noise = np.random.normal(0, 80, target.shape)
    pred = np.clip(target + noise, 0, 255).astype(np.uint8)

    print("gray_1: {}\tgray_2:{}\tdelta:{}".format(gray_1, gray_2, gray_1 - gray_2))
    print("noise a_mean:{}, std: {}".format(np.abs(noise).mean(), noise.std()))

    pixel_loss = F.l1_loss(torch.from_numpy(pred).unsqueeze(dim=0).to(torch.float32),
                           torch.from_numpy(target).unsqueeze(dim=0)).to(torch.float32)
    print("pixel loss: {:.4f}".format(pixel_loss))

    # Initialize the loss function
    criterion_grad = GradientMagnitudePhaseLoss(magnitude_weight=1.0, phase_weight=0.3)

    # Compute loss
    total_loss, loss_breakdown = criterion_grad(torch.from_numpy(pred).unsqueeze(dim=0).permute(0,3,1,2).to(torch.float32),
                                                torch.from_numpy(target).unsqueeze(dim=0).permute(0,3,1,2).to(torch.float32))
    print("Total gradient loss: {:.4f}".format(total_loss))
    print("Magnitude loss: {:.4f}".format(loss_breakdown['magnitude_loss']))
    print("Phase loss: {:.4f}".format(loss_breakdown['phase_loss']))

    cv2.imshow("noise image", pred)
    cv2.imshow("gt", target)
    cv2.waitKey()
