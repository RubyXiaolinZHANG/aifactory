import os
import torch
from aifactory.libs.nn.nets.unet_transformer import CnnTransformerUnet
from aifactory.libs.nn.blocks.cnn_transformer import CnnTransformerAttentionBlockParameters, \
    CnnTransformerMlpBlockParameters, CnnTransformerInteractBlockParameters
from aifactory.libs.nn.blocks import from_dict, BasicBlockParameters


BLOCK_PARAMETERS_DICT = {"attn_block_params": CnnTransformerAttentionBlockParameters,
                         "self_attn_block_params": CnnTransformerAttentionBlockParameters,
                         "cross_attn_block_params": CnnTransformerAttentionBlockParameters,
                         "mlp_params": CnnTransformerMlpBlockParameters,
                         "up_sample_params": CnnTransformerInteractBlockParameters,
                         "down_sample_params": CnnTransformerInteractBlockParameters}


def parse_unet_transformer_arc_params(unet_arc_params):
    if isinstance(unet_arc_params['heads'], BasicBlockParameters):
        return unet_arc_params
    encoder_parameters = unet_arc_params['encoder']
    for stage_name, stage_params in encoder_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(BLOCK_PARAMETERS_DICT[block_name], block_params)
    decoder_parameters = unet_arc_params['decoder']
    for stage_name, stage_params in decoder_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(BLOCK_PARAMETERS_DICT[block_name], block_params)
    head_parameters = unet_arc_params['heads']
    for head_name, head_params in head_parameters.items():
        head_parameters[head_name] = from_dict(BasicBlockParameters, head_params)
    return {'encoder': encoder_parameters,
            "decoder": decoder_parameters,
            "heads": head_parameters}


class AinrUnetTransformerSingleFrame(torch.nn.Module):

    def __init__(self, unet_arc_params):
        super().__init__()
        self.preprocess = torch.nn.PixelUnshuffle(2)
        self.denoise = CnnTransformerUnet(parse_unet_transformer_arc_params(unet_arc_params))
        self.channel2space = torch.nn.PixelShuffle(2)

    def forward(self, present, previous=None):
        preprocess = self.preprocess(present)
        net_out = self.denoise(preprocess)[0]
        mean_val = preprocess[:, :4].mean(dim=(2, 3)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        std_val = (preprocess[:, :4].var(dim=(2, 3)) + 1e-5).sqrt().unsqueeze(dim=-1).unsqueeze(dim=-1)
        denoise = preprocess + net_out * std_val + mean_val
        return {"denoise": self.channel2space(denoise),
                "fusion": self.channel2space(denoise),
                "noise": self.channel2space(net_out)}


if __name__ == "__main__":
    from torchinfo import summary
    import onnx
    from aifactory.utils.load_file import load_file

    INIT_MODEL_FROM_CONFIG = False
    yaml_file = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/configs/models/model_ainr_cnn_transformer_1t.yaml'
    config = load_file(yaml_file)

    if INIT_MODEL_FROM_CONFIG:
        unet_arc_parameters = config["parameters"]['arch']
    else:
        # get unet arch parameters
        encoder_parameters = config['parameters']['arch']['encoder']
        for stage_name, stage_params in encoder_parameters.items():
            for layer_name, layer_params in stage_params.items():
                for block_name, block_params in layer_params.items():
                    layer_params[block_name] = from_dict(BLOCK_PARAMETERS_DICT[block_name], block_params)
        decoder_parameters = config['parameters']['arch']['decoder']
        for stage_name, stage_params in decoder_parameters.items():
            for layer_name, layer_params in stage_params.items():
                for block_name, block_params in layer_params.items():
                    layer_params[block_name] = from_dict(BLOCK_PARAMETERS_DICT[block_name], block_params)
        head_parameters = config["parameters"]['arch']['heads']
        for head_name, head_params in head_parameters.items():
            head_parameters[head_name] = from_dict(BasicBlockParameters, head_params)
        unet_arc_parameters = {'encoder': encoder_parameters,
                               "decoder": decoder_parameters,
                               "heads": head_parameters}

    h, w = 1504 * 2, 2000 * 2
    dummy_input = {"present": torch.randn(1, 1, h, w),
                   "previous": torch.randn(1, 1, h, w)}

    model = AinrUnetTransformerSingleFrame(unet_arc_parameters)
    result = model(*list(dummy_input.values()))
    summary(model,
            input_data=(list(dummy_input.values())),
            verbose=2)

    onnx_file = './onnx/model_ainr_unet_transformer_single.onnx'
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(model, dummy_input, onnx_file, simplify=True, opset=13)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)

    onnx_file = './onnx/model_ainr_unet_transformer_single_backbone.onnx'
    torch.onnx.export(model.denoise, torch.randn(1, 4, h // 2, w // 2), onnx_file, simplify=True, opset=13)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)

    print("Done!")
