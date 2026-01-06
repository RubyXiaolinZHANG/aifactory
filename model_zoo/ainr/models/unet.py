import torch
import torch.nn as nn
import torch.nn.functional as F

class SpaceToDepth(torch.nn.Module):
    def __init__(self, block_size=2):
        super(SpaceToDepth, self).__init__()
        assert block_size in {
            2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        # (N, bs, bs, C, H//bs, W//bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def extra_repr(self):
        return f"block_size={self.block_size}"


class UnetBackbone(nn.Module):
    def __init__(self, ):
        super(UnetBackbone, self).__init__()

        self.init_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.encoder_1_1_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.down_1_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.encoder_2_1_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.down_2_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.encoder_3_1_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.down_3_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.bottom_1_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bottom_2_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.up_3_conv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.decoder_3_1_conv = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.up_2_conv = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.decoder_2_1_conv = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)

        self.up_1_conv = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.decoder_1_1_conv = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)

        self.last_conv = nn.Conv2d(in_channels=8, out_channels=5, kernel_size=3, padding=1)

    def forward(self, x_cur):
        feature_maps = []
        features = self.init_conv(x_cur)
        features = F.relu(features)
        feature_maps.append(features)

        features_en_1 = self.encoder_1_1_conv(features)
        features_en_1 = F.relu(features_en_1)
        feature_maps.append(features_en_1)
        features_down_1 = self.down_1_conv(features_en_1)
        features_down_1 = F.relu(features_down_1)

        features_en_2 = self.encoder_2_1_conv(features_down_1)
        features_en_2 = F.relu(features_en_2)
        feature_maps.append(features_en_2)
        features_down_2 = self.down_2_conv(features_en_2)
        features_down_2 = F.relu(features_down_2)

        features_en_3 = self.encoder_3_1_conv(features_down_2)
        features_en_3 = F.relu(features_en_3)
        feature_maps.append(features_en_3)
        features_down_3 = self.down_3_conv(features_en_3)
        features_down_3 = F.relu(features_down_3)

        features_bottom = self.bottom_1_conv(features_down_3)
        features_bottom = F.relu(features_bottom)
        features_bottom = self.bottom_2_conv(features_bottom)
        features_bottom = F.relu(features_bottom)
        feature_maps.append(features_bottom)

        features_up_3 = self.up_3_conv(features_bottom)
        features_up_3 = F.relu(features_up_3) # negtive impact
        features_de_3 = torch.cat([features_en_3, features_up_3], dim=1)
        features_de_3 = self.decoder_3_1_conv(features_de_3)
        features_de_3 = F.relu(features_de_3)
        feature_maps.append(features_de_3)

        features_up_2 = self.up_2_conv(features_de_3)
        features_up_2 = F.relu(features_up_2)
        features_de_2 = torch.cat([features_en_2, features_up_2], dim=1)
        features_de_2 = self.decoder_2_1_conv(features_de_2)
        features_de_2 = F.relu(features_de_2)
        feature_maps.append(features_de_2)

        features_up_1 = self.up_1_conv(features_de_2)
        features_up_1 = F.relu(features_up_1)
        features_de_1 = torch.cat([features_en_1, features_up_1], dim=1)
        features_de_1 = self.decoder_1_1_conv(features_de_1)
        features_de_1 = F.relu(features_de_1)
        feature_maps.append(features_de_1)

        features_out = self.last_conv(features_de_1)

        return features_out, feature_maps

class Unet(nn.Module):
    def __init__(self, ):
        super(Unet, self).__init__()

        self.pack = SpaceToDepth(block_size=2)
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.denoise = UnetBackbone()

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        ft0 = self.pack(x[:, 0:1, :, :])
        ft1 = self.pack(x[:, 1:2, :, :])

        input_pack_q = torch.cat([ft0, ft1], dim=1)
        output, feature_maps = self.denoise(input_pack_q)

        mask = output[:, 4:5, :, :]
        gamma = F.sigmoid(mask)
        denoise_out = output[:, 0:4, ...] + input_pack_q[:, 4:8, ...]
        final_out = (1 - gamma) * input_pack_q[:, 0:4, ...] + gamma * denoise_out

        denoise_out = self.up(denoise_out)
        final_out = self.up(final_out)

        return gamma, denoise_out, final_out, feature_maps


if __name__ == "__main__":
    from thop import profile
    model = UnetBackbone().to('cpu')
    input0 = torch.randn(1, 8, 1504, 2000)
    macs1, params1 = profile(model, inputs=(input0, ))
    print(model)
    print('GOPs = ' + str(macs1 * 2 / 1000 ** 3) + 'G')
    print('Params = ' + str(params1 / 1000 ** 1) + 'K')