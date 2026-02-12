import torch
from aifactory.libs.nn.modules.demosaic import Demosaic
from aifactory.libs.loss.hsv_loss import HSVLoss


class RawColorBindingLoss(torch.nn.Module):

    def __init__(self, pattern, phase_weight=1.0, contrast_weight=1.0, hsv_weight=None, loss=None):
        super().__init__()
        self._phase_weight = phase_weight
        self._contrast_weight = contrast_weight
        self._pattern = pattern
        self._demosaic = Demosaic()
        if loss is None:
            self._loss = torch.nn.L1Loss()
        else:
            self._loss = eval(loss)()
        if hsv_weight is None:
            hsv_weight = [1.0, 1.0, 0]
        self._hsv_loss = HSVLoss( h_weight=hsv_weight[0], s_weight=hsv_weight[1], v_weight=hsv_weight[2], loss=loss)

    def __call__(self, predict, target):
        predict_rggb = self._demosaic.bilinear_interpolate(predict, pattern=self._pattern.upper())
        predict_rgb_p = self.get_rgb_potential(predict_rggb)
        target_rggb = self._demosaic.bilinear_interpolate(target, pattern=self._pattern.upper())
        target_rgb_p = self.get_rgb_potential(target_rggb)
        rgb_loss = self._loss(predict_rgb_p, target_rgb_p)
        predict_g_contrast = (predict_rggb[:, 1, :, :] - predict_rggb[:, 2, :, :]) / (
                predict_rggb[:, 1, :, :] + predict_rggb[:, 2, :, :] + 1e-8)
        target_g_contrast = (target_rggb[:, 1, :, :] - target_rggb[:, 2, :, :]) / (
                target_rggb[:, 1, :, :] + target_rggb[:, 2, :, :] + 1e-8)
        g_loss = self._loss(predict_g_contrast, target_g_contrast)
        loss = self._phase_weight * rgb_loss + self._contrast_weight * g_loss + self._hsv_loss(predict_rgb_p, target_rgb_p)
        return loss

    def get_rgb_potential(self, rggb):
        rgb = torch.concat([rggb[:, 0, :, :].unsqueeze(dim=1),
                            ((rggb[:, 1, :, :] + rggb[:, 2, :, :]) / 2).unsqueeze(dim=1),
                            rggb[:, 3, :, :].unsqueeze(dim=1)], dim=1)
        s = torch.linalg.norm(rgb, ord=2, dim=1, keepdim=True)
        p = rgb / s
        return p


if __name__ == "__main__":
    import cv2
    import numpy as np
    from aifactory.libs.common.mem_ops import depth2space

    image_file = r"D:\Program\ToGit\xiaomi\aifactory\tests\images\vimeo_00001_0003_im1.png"
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    raw = np.array([image[::2, ::2, 2],  # r
                    image[::2, 1::2, 1],  # g
                    image[1::2, ::2, 1],  # g
                    image[1::2, 1::2, 0]])  # b
    noise_image = (image.astype(np.float32) + np.random.normal(0, 50, image.shape)).round().astype(np.uint8)
    sensor_raw = np.array([noise_image[::2, ::2, 2],  # r
                           noise_image[::2, 1::2, 1],  # g
                           noise_image[1::2, ::2, 1],  # g
                           noise_image[1::2, 1::2, 0]])  # b

    raw = depth2space(raw, block_size=2, channels_last=False)
    sensor_raw = depth2space(sensor_raw, block_size=2, channels_last=False)
    print("raw shape: {}".format(raw.shape))

    cv2.imwrite("image.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imwrite("noise_image.png", cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite("rggb.png", raw)
    cv2.imwrite("rggb_noise.png", sensor_raw)

    raw = torch.from_numpy(np.transpose(raw.astype(np.float32), [2, 0, 1])).unsqueeze(dim=0)
    noise_raw = torch.from_numpy(np.transpose(sensor_raw.astype(np.float32), [2, 0, 1])).unsqueeze(dim=0)
    noise_raw.requires_grad_(True)
    print("raw tensor shape: {}".format(raw.shape))

    color_loss = RawColorBindingLoss(pattern="rggb")
    loss = color_loss(noise_raw, raw)
    print(loss)
    loss.backward()

'''

image = (target_rgb_p.squeeze().permute(1,2,0).detach().numpy()*255).astype(np.uint8)

image = ((s/s.max()).squeeze().detach().numpy()*255).astype(np.uint8)

image = ((rgb/rgb.max()).squeeze().permute(1,2,0).detach().numpy()*255).astype(np.uint8)

image = ((p/p.max()).squeeze().permute(1,2,0).detach().numpy()*255).astype(np.uint8)

image = ((target_rgb_p/target_rgb_p.max()).squeeze().permute(1,2,0).detach().numpy()*255).astype(np.uint8)

image = (((predict_g_contrast + 1)/2).squeeze().detach().numpy()*255).astype(np.uint8)
'''
