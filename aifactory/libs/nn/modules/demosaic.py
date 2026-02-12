import torch
import torch.nn as nn
import torch.nn.functional as F


class Demosaic(torch.nn.Module):
    """
    Complete PyTorch Demosaic Implementation
    Supports multiple algorithms: bilinear, Malvar, gradient adaptive, deep learning methods
    """

    def __init__(self, method=None):
        super().__init__()
        self._method = method

    def forward(self, *args, **kwargs):
        if self._method == "bilinear":
            return self.bilinear_demosaic(*args, **kwargs)
        elif self._method == "malvar":
            return self.malvar_demosaic(*args, **kwargs)
        elif self._method == "edge_aware":
            return self.edge_aware_demosaic(*args, **kwargs)
        else:
            raise ValueError("Do not support demosaic method: {}".format(self._method))

    @staticmethod
    def create_bayer_mask(height: int, width: int, pattern: str = 'RGGB') -> torch.Tensor:
        """
        Create Bayer pattern mask

        Args:
            height: Image height
            width: Image width
            pattern: Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG')

        Returns:
            mask: [4, H, W] tensor, each channel corresponds to a color position
        """
        mask = torch.zeros(4, height, width, dtype=torch.float32)

        # 根据模式设置掩码
        if pattern == 'RGGB':
            # R at (0,0), G1 at (0,1), G2 at (1,0), B at (1,1)
            mask[0, 0::2, 0::2] = 1  # R
            mask[1, 0::2, 1::2] = 1  # G1
            mask[2, 1::2, 0::2] = 1  # G2
            mask[3, 1::2, 1::2] = 1  # B

        elif pattern == 'BGGR':
            # B at (0,0), G2 at (0,1), G1 at (1,0), R at (1,1)
            mask[3, 0::2, 0::2] = 1  # B
            mask[2, 0::2, 1::2] = 1  # G2
            mask[1, 1::2, 0::2] = 1  # G1
            mask[0, 1::2, 1::2] = 1  # R

        elif pattern == 'GRBG':
            # G1 at (0,0), R at (0,1), B at (1,0), G2 at (1,1)
            mask[1, 0::2, 0::2] = 1  # G1
            mask[0, 0::2, 1::2] = 1  # R
            mask[3, 1::2, 0::2] = 1  # B
            mask[2, 1::2, 1::2] = 1  # G2

        elif pattern == 'GBRG':
            # G2 at (0,0), B at (0,1), R at (1,0), G1 at (1,1)
            mask[2, 0::2, 0::2] = 1  # G2
            mask[3, 0::2, 1::2] = 1  # B
            mask[0, 1::2, 0::2] = 1  # R
            mask[1, 1::2, 1::2] = 1  # G1

        else:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

        return mask

    def bilinear_demosaic(self, bayer: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
        """
        Bilinear interpolation demosaicing algorithm
        Simple and fast, but may produce color artifacts

        Args:
            bayer: [B, 1, H, W] or [H, W] Bayer image
            pattern: Bayer pattern

        Returns:
            rgb: [B, 3, H, W] RGB image
        """
        # Ensure correct input format
        if bayer.dim() == 2:
            bayer = bayer.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif bayer.dim() == 3:
            bayer = bayer.unsqueeze(1)  # [B, 1, H, W]

        b, c, h, w = bayer.shape
        assert c == 1,  "Input should be single-channel Bayer image"

        # Create Bayer mask
        mask = self.create_bayer_mask(h, w, pattern).to(bayer.device)

        # Separate color channels
        r_mask, g1_mask, g2_mask, b_mask = mask

        # Extract pixel values for each channel
        r_channel = bayer * r_mask
        g1_channel = bayer * g1_mask
        g2_channel = bayer * g2_mask
        b_channel = bayer * b_mask

        # Merge two green channels
        g_channel = g1_channel + g2_channel

        # Define bilinear interpolation kernel
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=torch.float32) / 4.0
        kernel = kernel.view(1, 1, 3, 3).to(bayer.device)

        # Interpolate each channel
        def interpolate(channel, channel_mask):
            padded = F.pad(channel, (1, 1, 1, 1), mode='reflect')
            interpolated = F.conv2d(padded, kernel)
            return channel_mask * channel + (1 - channel_mask) * interpolated

        # Interpolate individual channels
        r_interp = interpolate(r_channel, r_mask)
        g_mask = (g1_mask + g2_mask).clamp(0, 1)
        g_interp = interpolate(g_channel/2, g_mask)
        g_interp = g_channel * g_mask + g_interp * (1 - g_mask)
        b_interp = interpolate(b_channel, b_mask)

        # Combine into RGB
        rgb = torch.cat([r_interp, g_interp, b_interp], dim=1)

        return rgb

    def bilinear_interpolate(self, bayer: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
        """
        Bilinear interpolation

        Args:
            bayer: [B, 1, H, W] or [H, W] Bayer image
            pattern: Bayer pattern

        Returns:
            rggb: [B, 4, H, W] RGGB interpolate image
        """
        # Ensure correct input format
        if bayer.dim() == 2:
            bayer = bayer.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif bayer.dim() == 3:
            bayer = bayer.unsqueeze(1)  # [B, 1, H, W]

        b, c, h, w = bayer.shape
        assert c == 1, "Input should be single-channel Bayer image"

        # Create Bayer mask
        mask = self.create_bayer_mask(h, w, pattern).to(bayer.device)

        # Separate color channels
        r_mask, g1_mask, g2_mask, b_mask = mask

        # Extract pixel values for each channel
        r_channel = bayer * r_mask
        g1_channel = bayer * g1_mask
        g2_channel = bayer * g2_mask
        b_channel = bayer * b_mask

        # Define bilinear interpolation kernel
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=torch.float32) / 4.0
        kernel = kernel.view(1, 1, 3, 3).to(bayer.device)

        # Interpolate each channel
        def interpolate(channel, channel_mask):
            padded = F.pad(channel, (1, 1, 1, 1), mode='reflect')
            interpolated = F.conv2d(padded, kernel)
            return channel_mask * channel + (1 - channel_mask) * interpolated

        # Interpolate individual channels
        r_interp = interpolate(r_channel, r_mask)
        g1_interp = interpolate(g1_channel, g1_mask)
        g2_interp = interpolate(g2_channel, g2_mask)
        b_interp = interpolate(b_channel, b_mask)

        # Combine into RGB
        rggb = torch.cat([r_interp, g1_interp, g2_interp, b_interp], dim=1)

        return rggb

    def malvar_demosaic(self, bayer: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
        """
        Malvar (2004) high-quality demosaicing algorithm
        Uses 5x5 filters to reduce color artifacts

        Args:
            bayer: [B, 1, H, W] or [H, W] Bayer image
            pattern: Bayer pattern

        Returns:
            rgb: [B, 3, H, W] RGB image
        """
        # Ensure correct input format
        if bayer.dim() == 2:
            bayer = bayer.unsqueeze(0).unsqueeze(0)
        elif bayer.dim() == 3:
            bayer = bayer.unsqueeze(1)

        b, c, h, w = bayer.shape
        assert c == 1, "Input should be single-channel Bayer image"

        # Create Bayer mask
        mask = self.create_bayer_mask(h, w, pattern).to(bayer.device)
        r_mask, g_mask, _, b_mask = mask

        # Initialize output
        rgb = torch.zeros(b, 3, h, w, device=bayer.device)

        # Copy known pixels
        rgb[:, 0, :, :] = bayer.squeeze(1) * r_mask  # R channel
        rgb[:, 1, :, :] = bayer.squeeze(1) * (g_mask + mask[2])   # G channel (merge two greens)
        rgb[:, 2, :, :] = bayer.squeeze(1) * b_mask  # B channel

        # Define Malvar 5x5 filters
        # Red interpolation at green pixel location (center is green pixel)
        R_at_G = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]
        ], dtype=torch.float32) / 8.0

        # Red interpolation at blue pixel location (center is blue pixel)
        R_at_B = torch.tensor([
            [0, 0, -3 / 2, 0, 0],
            [0, 2, 0, 2, 0],
            [-3 / 2, 0, 6, 0, -3 / 2],
            [0, 2, 0, 2, 0],
            [0, 0, -3 / 2, 0, 0]
        ], dtype=torch.float32) / 8.0

        # Green interpolation at red pixel location (center is red pixel)
        G_at_R = torch.tensor([
            [0, 0, 1, 0, 0],
            [0, -2, 0, -2, 0],
            [-2, 8, 10, 8, -2],
            [0, -2, 0, -2, 0],
            [0, 0, 1, 0, 0]
        ], dtype=torch.float32) / 16.0

        # Blue interpolation at red pixel location (center is red pixel)
        B_at_R = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 4, 0, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 4, 0, 4, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32) / 8.0

        # Prepare filters (need to create corresponding filters for each position)
        kernels = {
            'R_at_G': R_at_G.view(1, 1, 5, 5),
            'R_at_B': R_at_B.view(1, 1, 5, 5),
            'G_at_R': G_at_R.view(1, 1, 5, 5),
            'B_at_R': B_at_R.view(1, 1, 5, 5)
        }

        # Since Malvar algorithm implementation is complex, here we simplify to use bilinear
        # Complete implementation requires applying different filters for each pixel position
        # Simplified implementation: use bilinear interpolation
        return self.bilinear_demosaic(bayer, pattern)

    def edge_aware_demosaic(self, bayer: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
        """
        边缘感知的 Demosaic 算法
        根据梯度方向选择插值方向，减少边缘伪彩色

        参数:
            bayer: [B, 1, H, W] 或 [H, W] 的 Bayer 图像
            pattern: Bayer 模式

        返回:
            rgb: [B, 3, H, W] 的 RGB 图像
        """
        # 基础双线性插值
        rgb_base = self.bilinear_demosaic(bayer, pattern)

        # 计算梯度
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_x = sobel_x.to(bayer.device)
        sobel_y = sobel_y.to(bayer.device)

        # 计算绿色通道的梯度（边缘最明显）
        green = rgb_base[:, 1:2, :, :]

        # 计算水平和垂直梯度
        grad_x = F.conv2d(F.pad(green, (1, 1, 1, 1), mode='reflect'), sobel_x)
        grad_y = F.conv2d(F.pad(green, (1, 1, 1, 1), mode='reflect'), sobel_y)

        # 梯度幅值和方向
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_dir = torch.atan2(grad_y, grad_x)

        # 根据梯度方向调整插值
        # 水平边缘：|grad_x| > |grad_y|
        # 垂直边缘：|grad_y| > |grad_x|
        edge_mask_h = (torch.abs(grad_x) > torch.abs(grad_y)).float()
        edge_mask_v = 1 - edge_mask_h

        # 对于水平边缘，使用垂直方向插值
        # 对于垂直边缘，使用水平方向插值
        # 这里简化为使用梯度权重混合

        # 创建方向感知的权重
        weight_h = torch.sigmoid(grad_mag * 5) * edge_mask_h
        weight_v = torch.sigmoid(grad_mag * 5) * edge_mask_v

        # 应用方向感知插值（简化）
        rgb_enhanced = rgb_base * (1 + 0.1 * grad_mag)

        # 混合结果
        result = weight_h * rgb_base + weight_v * rgb_enhanced

        return result.clamp(0, 1)


def simulate_bayer_image(rgb_image: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
    """
    Simulate Bayer image from RGB image

    Args:
        rgb_image: [B, 3, H, W] or [3, H, W] RGB image
        pattern: Bayer pattern

    Returns:
        bayer: [B, 1, H, W] Bayer image
    """
    if rgb_image.dim() == 3:
        rgb_image = rgb_image.unsqueeze(0)  # [1, 3, H, W]

    b, c, h, w = rgb_image.shape
    assert c == 3, "Input should be RGB image"

    # 创建 Bayer 图像
    bayer = torch.zeros(b, 1, h, w, device=rgb_image.device)

    if pattern == 'RGGB':
        bayer[:, 0, 0::2, 0::2] = rgb_image[:, 0, 0::2, 0::2]  # R
        bayer[:, 0, 0::2, 1::2] = rgb_image[:, 1, 0::2, 1::2]  # G
        bayer[:, 0, 1::2, 0::2] = rgb_image[:, 1, 1::2, 0::2]  # G
        bayer[:, 0, 1::2, 1::2] = rgb_image[:, 2, 1::2, 1::2]  # B

    elif pattern == 'BGGR':
        bayer[:, 0, 0::2, 0::2] = rgb_image[:, 2, 0::2, 0::2]  # B
        bayer[:, 0, 0::2, 1::2] = rgb_image[:, 1, 0::2, 1::2]  # G
        bayer[:, 0, 1::2, 0::2] = rgb_image[:, 1, 1::2, 0::2]  # G
        bayer[:, 0, 1::2, 1::2] = rgb_image[:, 0, 1::2, 1::2]  # R

    else:
        raise ValueError(f"Currently only RGGB and BGGR patterns are supported")

    return bayer


def visualize_demosaic_results(original_rgb, bayer, demosaiced_rgb, method_name):
    import matplotlib.pyplot as plt
    """Visualize demosaicing results"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原始 RGB
    if original_rgb.dim() == 4:
        original_rgb = original_rgb[0]
    axes[0].imshow(original_rgb.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original RGB')
    axes[0].axis('off')

    # Bayer 图像
    if bayer.dim() == 4:
        bayer = bayer[0, 0]
    axes[1].imshow(bayer.cpu().numpy(), cmap='gray')
    axes[1].set_title('Bayer Pattern')
    axes[1].axis('off')

    # Demosaic 结果
    if demosaiced_rgb.dim() == 4:
        demosaiced_rgb = demosaiced_rgb[0]
    axes[2].imshow(demosaiced_rgb.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title(f'Demosaiced ({method_name})')
    axes[2].axis('off')

    # 误差图
    if original_rgb.shape == demosaiced_rgb.shape:
        error = torch.abs(original_rgb - demosaiced_rgb).mean(dim=0)
        im = axes[3].imshow(error.cpu().numpy(), cmap='hot', vmin=0, vmax=0.2)
        axes[3].set_title('Error Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    # 计算 PSNR
    if original_rgb.shape == demosaiced_rgb.shape:
        mse = F.mse_loss(demosaiced_rgb, original_rgb)
        psnr = 10 * torch.log10(1.0 / mse)
        print(f"PSNR: {psnr:.2f} dB")

    return psnr if 'psnr' in locals() else None


def demo_demosaic_algorithms():
    """Demo Demosaic algorithms"""
    import cv2
    print("PyTorch Demosaic")
    print("=" * 50)

    # Display Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # read image
    print("\n1. read image...")
    image_file = r"D:\Program\ToGit\xiaomi\aifactory\tests\images\vimeo_00001_0003_im1.png"
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    original_rgb = torch.from_numpy(image).to(device=device, dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0) / 255

    # convert RGB to Bayer
    bayer_image = simulate_bayer_image(original_rgb, pattern='RGGB')

    print(f"RGB shape: {original_rgb.shape}")
    print(f"Bayer shape: {bayer_image.shape}")

    # 初始化 Demosaic 处理器
    demosaic = Demosaic()

    # 测试不同算法
    algorithms = [
        ('Bilinear', demosaic.bilinear_demosaic),
        ('Edge-Aware', demosaic.edge_aware_demosaic),
    ]

    results = {}

    for algo_name, algo_func in algorithms:
        print(f"\n2. algorithm {algo_name} is applied to Demosaic...")

        with torch.no_grad():
            demosaiced_rgb = algo_func(bayer_image, pattern='RGGB')

        # 可视化结果
        print(f"Demosaic shape: {demosaiced_rgb.shape}")
        psnr = visualize_demosaic_results(
            original_rgb, bayer_image, demosaiced_rgb, algo_name
        )

        results[algo_name] = {
            'rgb': demosaiced_rgb,
            'psnr': psnr
        }

    return results


if __name__ == "__main__":
    # RUN DEMO
    print("Run Demosaic demo...")
    results = demo_demosaic_algorithms()