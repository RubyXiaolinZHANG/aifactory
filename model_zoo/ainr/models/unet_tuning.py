import os

import torch
DEFAULT_ADAPTOR = {"input_chanels": [8]}

DEFAULT_ENCODER_STAGE_1 = {"input_chanels": [8]}

DEFAULT_ARC = {"ad"}



class Preprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.space2depth = torch.nn.PixelUnshuffle(2)

    def forward(self, present, previous):
        return self.space2depth(torch.concat([present, previous], dim=1))

class UnetBackboneWithParamTuning(torch.nn.Module):

    def __init__(self, base_width=8, stage_x_w=None, stage_x_d=None):
        super(UnetBackboneWithParamTuning, self).__init__()
        self._encoder = []
        # self._mid_layer =

    def forward(self, x_cur):
        features = self.adaptor(x_cur)
        features = torch.relu(features)

        features_en_1 = self.encoder_1_1_conv(features)
        features_en_1 = torch.relu(features_en_1)
        features_down_1 = self.down_1_conv(features_en_1)
        features_down_1 = torch.relu(features_down_1)

        features_en_2 = self.encoder_2_1_conv(features_down_1)
        features_en_2 = torch.relu(features_en_2)
        features_down_2 = self.down_2_conv(features_en_2)
        features_down_2 = torch.relu(features_down_2)

        features_en_3 = self.encoder_3_1_conv(features_down_2)
        features_en_3 = torch.relu(features_en_3)
        features_down_3 = self.down_3_conv(features_en_3)
        features_down_3 = torch.relu(features_down_3)

        features_bottom = self.bottom_1_conv(features_down_3)
        features_bottom = torch.relu(features_bottom)
        features_bottom = self.bottom_2_conv(features_bottom)
        features_bottom = torch.relu(features_bottom)

        features_up_3 = self.up_3_conv(features_bottom)
        features_up_3 = torch.relu(features_up_3)
        features_de_3 = torch.cat([features_en_3, features_up_3], dim=1)
        features_de_3 = self.decoder_3_1_conv(features_de_3)
        features_de_3 = torch.relu(features_de_3)

        features_up_2 = self.up_2_conv(features_de_3)
        features_up_2 = torch.relu(features_up_2)
        features_de_2 = torch.cat([features_en_2, features_up_2], dim=1)
        features_de_2 = self.decoder_2_1_conv(features_de_2)
        features_de_2 = torch.relu(features_de_2)

        features_up_1 = self.up_1_conv(features_de_2)
        features_up_1 = torch.relu(features_up_1)
        features_de_1 = torch.cat([features_en_1, features_up_1], dim=1)
        features_de_1 = self.decoder_1_1_conv(features_de_1)
        features_de_1 = torch.relu(features_de_1)

        features_out = self.head(features_de_1)

        return features_out


class Postprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.channel2space = torch.nn.PixelShuffle(2)

    def forward(self, present, previous, noise, weight):
        denoise = present + noise
        score = torch.sigmoid(weight)
        fusion = denoise * score + previous * (1 - score)
        return {"denoise": self.channel2space(denoise),
                "fusion": self.channel2space(fusion),
                "noise": self.channel2space(noise),
                "fusion_mask": score}
    

class AinrUnetWithParamTuning(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.preprocess = Preprocess()
        self.denoise = UnetBackboneWithParamTuning()
        self.postprocess = Postprocess()

    def forward(self, present, previous):
        preprocess = self.preprocess(present, previous)
        net_out = self.denoise(preprocess)
        result = self.postprocess(preprocess[:, :4], preprocess[:, 4:], net_out[:,:4], net_out[:,4].unsqueeze(dim=1))
        return result


if __name__ == "__main__":
    from torchinfo import summary
    import onnx
    from aifactory.utils.load_file import load_file
    config_file = "D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/configs/AINR_Unet_Tuning.yaml"
    config = load_file(config_file)

    h, w = 1504 * 2, 2000 * 2
    dummy_input = {"present": torch.randn(1, 1, h, w),
                   "previous": torch.randn(1, 1, h, w)}

    model = AinrUnetWithParamTuning()
    summary(model,
            input_data=(list(dummy_input.values())),
            verbose=2)

    onnx_file = './onnx/ainr_unet_.onnx'
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(model, dummy_input, onnx_file, simplify=True, opset=13)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)

    onnx_file = './onnx/ainr_unet_backbone_.onnx'
    torch.onnx.export(model.denoise, torch.randn(1, 8, h//2, w//2), onnx_file, simplify=True, opset=13)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)