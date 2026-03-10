import torch
from aifactory.libs.nn.blocks.cnn_transformer import CnnTransformerEncoderBlock, CnnTransformerDecoderBlock
from aifactory.libs.nn.blocks import BasicBlockParameters
from aifactory.libs.nn.nets.unet import UnetHead

def build_cnn_transformer_encoder_stage(cls, stages_parameters):
    layers = []
    for stage_name, stage_params in stages_parameters.items():
        setattr(cls, stage_name, CnnTransformerEncoderBlock(stage_params['self_attn_block_params'],
                                                            stage_params['mlp_params'],
                                                            stage_params['down_sample_params']))
        layers.append(getattr(cls, stage_name))
    setattr(cls, "_layers", layers)


class CnnTransformerEncoderStage(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_cnn_transformer_encoder_stage(self, stages_parameters)

    def forward(self, x):
        for layer_id, layer in enumerate(self._layers):
            x, kv = layer(x)
        return x, kv


def build_cnn_transformer_encoder(cls, stages_parameters):
    stages = []
    for stage_name, stage_params in stages_parameters.items():
        if 'attn_block_params' in stage_params:
            setattr(cls, stage_name, CnnTransformerEncoderBlock(stage_params['self_attn_block_params'],
                                                                stage_params['mlp_params'],
                                                                stage_params['down_sample_params']))
        else:
            setattr(cls, stage_name, CnnTransformerEncoderStage(stage_params))

        stages.append(getattr(cls, stage_name))
    setattr(cls, "_stages", stages)


class CnnTransformerEncoder(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_cnn_transformer_encoder(self, stages_parameters)

    def forward(self, x):
        kv_catch = {}
        for stage_id, stage in enumerate(self._stages):
            x, kv = stage(x)
            kv.update({"output": x})
            kv_catch["stage_{}".format(stage_id)] = kv
        return x, kv_catch  # list(skip_connections.values())


def build_cnn_transformer_decoder_stage(cls, stages_parameters):
    layers = []
    for stage_name, stage_params in stages_parameters.items():
        setattr(cls, stage_name, CnnTransformerDecoderBlock(stage_params.get('self_attn_block_params', None),
                                                                stage_params.get('cross_attn_block_params', None),
                                                                stage_params['mlp_params'],
                                                                stage_params.get('up_sample_params', None)))
        layers.append(getattr(cls, stage_name))
    setattr(cls, "_layers", layers)


class CnnTransformerDecoderStage(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_cnn_transformer_decoder_stage(self, stages_parameters)

    def forward(self, x, k, v):
        for layer_id, layer in enumerate(self._layers):
            x = layer(x, k, v)
        return x


def build_cnn_transformer_decoder(cls, stages_parameters):
    stages = []
    for stage_name, stage_params in stages_parameters.items():
        if 'mlp_params' in stage_params:
            setattr(cls, stage_name, CnnTransformerDecoderBlock(stage_params.get('self_attn_block_params', None),
                                                                stage_params.get('cross_attn_block_params', None),
                                                                stage_params['mlp_params'],
                                                                stage_params.get('up_sample_params', None)))
        else:
            setattr(cls, stage_name, CnnTransformerDecoderStage(stage_params))

        stages.append(getattr(cls, stage_name))
    setattr(cls, "_stages", stages)


class CnnTransformerDecoder(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_cnn_transformer_decoder(self, stages_parameters)

    def forward(self, x, kv_catch):
        if isinstance(kv_catch, dict):
            kv_catch = list(kv_catch.values())
        elif isinstance(kv_catch, tuple):
            kv_catch = list(kv_catch)
        elif isinstance(kv_catch, list):
            pass
        else:
            raise ValueError("Only support skip connection type of dict, tuple list, but this is {}".format(
                type(kv_catch)))
        kv_catch.reverse()
        for stage_id, stage in enumerate(self._stages):
            # print("decoder stage: {}".format(stage_id))
            x = stage(x, kv_catch[stage_id]["k"],  kv_catch[stage_id]["v"])
        return x


class CnnTransformerUnet(torch.nn.Module):

    def __init__(self, unet_arc_parameters):
        super().__init__()
        self.encoder = CnnTransformerEncoder(unet_arc_parameters['encoder'])
        self.decoder = CnnTransformerDecoder(unet_arc_parameters['decoder'])
        self.heads = UnetHead(unet_arc_parameters['heads'])

    def forward(self, x):
        encoder, kv_catch = self.encoder(x)
        decoder = self.decoder(encoder, kv_catch)
        outputs = self.heads(decoder)
        return outputs


########################################################################################################################
# test codes
########################################################################################################################


def test_transformer_encoder(x):
    import os, cv2
    from torchinfo import summary
    from aifactory.libs.nn.blocks.cnn_transformer import CnnTransformerAttentionBlockParameters, \
        CnnTransformerMlpBlockParameters, CnnTransformerInteractBlockParameters
    from aifactory.libs.nn.blocks import from_dict
    from aifactory.utils.load_file import load_file

    # setting
    block_parameters = {"self_attn_block_params": CnnTransformerAttentionBlockParameters,
                        "mlp_params": CnnTransformerMlpBlockParameters,
                        "up_sample_params": CnnTransformerInteractBlockParameters,
                        "down_sample_params": CnnTransformerInteractBlockParameters}

    # prepare inputs
    yaml_file = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/configs/models/model_ainr_cnn_transformer.yaml'
    config = load_file(yaml_file)

    # init encoder
    stages_parameters = config['parameters']['arch']['encoder']
    for stage_name, stage_params in stages_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(block_parameters[block_name], block_params)
    encoder = CnnTransformerEncoder(stages_parameters)

    y, kv_catch = encoder(x)
    print("x: {}".format(x.shape))
    print("y: {}".format(y.shape))
    summary(encoder, input_data=x)
    onnx_file = os.path.join('../../../../onnx/cnn_transformer_encoder.onnx', )
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(encoder, x, onnx_file,
                      opset_version=11,
                      training=torch.onnx.TrainingMode.EVAL)
    print("save onns to: {}".format(os.path.abspath(onnx_file)))
    return y, kv_catch


def test_transformer_decoder(y, kv_catch):

    from aifactory.libs.nn.blocks.cnn_transformer import CnnTransformerAttentionBlockParameters, \
        CnnTransformerMlpBlockParameters, CnnTransformerInteractBlockParameters
    from aifactory.libs.nn.blocks import from_dict
    from aifactory.utils.load_file import load_file

    block_parameters = {"self_attn_block_params": CnnTransformerAttentionBlockParameters,
                        "cross_attn_block_params": CnnTransformerAttentionBlockParameters,
                        "mlp_params": CnnTransformerMlpBlockParameters,
                        "up_sample_params": CnnTransformerInteractBlockParameters,
                        "down_sample_params": CnnTransformerInteractBlockParameters}

    # prepare parameters
    yaml_file = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/configs/models/model_ainr_cnn_transformer.yaml'
    config = load_file(yaml_file)
    stages_parameters = config['parameters']['arch']['decoder']
    for stage_name, stage_params in stages_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(block_parameters[block_name], block_params)

    decoder = CnnTransformerDecoder(stages_parameters)
    print(decoder)
    x_hat = decoder(y, kv_catch)
    print("x_hat: {}".format(x_hat.shape))


def test_transformer_sep():
    import os, cv2
    from torchinfo import summary
    from aifactory.libs.nn.blocks.cnn_transformer import CnnTransformerAttentionBlockParameters, \
        CnnTransformerMlpBlockParameters, CnnTransformerInteractBlockParameters
    from aifactory.libs.nn.blocks import from_dict
    from aifactory.utils.load_file import load_file

    # config
    block_parameters = {"attn_block_params": CnnTransformerAttentionBlockParameters,
                        "mlp_params": CnnTransformerMlpBlockParameters,
                        "up_sample_params": CnnTransformerInteractBlockParameters,
                        "down_sample_params": CnnTransformerInteractBlockParameters}
    # prepare inputs
    yaml_file = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/configs/models/model_ainr_cnn_transformer.yaml'
    config = load_file(yaml_file)
    image = cv2.imread("../../../../tests/images/llama.jpg")
    h, w, _ = image.shape
    x = torch.from_numpy(image[:h//8*8, :w//8*8]).permute(2, 0, 1).unsqueeze(dim=0).to(torch.float32)

    # init encoder
    stages_parameters = config['parameters']['arch']['encoder']
    for stage_name, stage_params in stages_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(block_parameters[block_name], block_params)
    encoder = CnnTransformerEncoder(stages_parameters)

    # init decoder
    stages_parameters = config['parameters']['arch']['decoder']
    for stage_name, stage_params in stages_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(block_parameters[block_name], block_params)
    decoder = CnnTransformerDecoder(stages_parameters)
    print(decoder)
    #

    enc, skip_connections = encoder(x)
    dec = decoder(enc, skip_connections)




    print("x: {}".format(x.shape))
    print("enc: {}".format(enc.shape))
    print("dec: {}".format(dec.shape))


def test_transformer_unet(x):
    import os, cv2, onnx
    from torchinfo import summary
    from aifactory.libs.nn.blocks.cnn_transformer import CnnTransformerAttentionBlockParameters, \
        CnnTransformerMlpBlockParameters, CnnTransformerInteractBlockParameters
    from aifactory.libs.nn.blocks import from_dict
    from aifactory.utils.load_file import load_file

    # config
    block_parameters = {"attn_block_params": CnnTransformerAttentionBlockParameters,
                        "self_attn_block_params": CnnTransformerAttentionBlockParameters,
                        "cross_attn_block_params": CnnTransformerAttentionBlockParameters,
                        "mlp_params": CnnTransformerMlpBlockParameters,
                        "up_sample_params": CnnTransformerInteractBlockParameters,
                        "down_sample_params": CnnTransformerInteractBlockParameters}
    # prepare inputs
    yaml_file = 'D:/Program/ToGit/xiaomi/aifactory/model_zoo/ainr/configs/models/model_ainr_cnn_transformer.yaml'
    config = load_file(yaml_file)
    encoder_parameters = config['parameters']['arch']['encoder']
    for stage_name, stage_params in encoder_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(block_parameters[block_name], block_params)
    decoder_parameters = config['parameters']['arch']['decoder']
    for stage_name, stage_params in decoder_parameters.items():
        for layer_name, layer_params in stage_params.items():
            for block_name, block_params in layer_params.items():
                layer_params[block_name] = from_dict(block_parameters[block_name], block_params)
    head_parameters = config["parameters"]['arch']['heads']
    for head_name, head_params in head_parameters.items():
        head_parameters[head_name] = from_dict(BasicBlockParameters, head_params)
    unet_arc_parameters = {'encoder': encoder_parameters,
                           "decoder": decoder_parameters,
                           "heads": head_parameters}
    unet = CnnTransformerUnet(unet_arc_parameters)

    y = unet(x)
    summary(unet, input_data=x)

    print("x:\tshape={}\tmin={}\tmax={}".format(x.shape, x.min(), x.max()))
    print("y:\tshape={}\tmin={}\tmax={}\tmean={}\tstd={}".format(y[0].shape, y[0].min(), y[0].max(), y[0].mean(), y[0].std()))
    onnx_file = os.path.join('../../../../onnx/ainr_cnn_transformer.onnx', )
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(unet, x, onnx_file,
                      opset_version=11,
                      training=torch.onnx.TrainingMode.EVAL)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)
    print("save onnx to: {}".format(os.path.abspath(onnx_file)))



if __name__ == "__main__":
    import cv2
    image = cv2.imread("../../../../tests/images/llama.jpg")
    h, w, _ = image.shape
    x = torch.from_numpy(image[:h // 8 * 8, :w // 8 * 8]).permute(2, 0, 1).unsqueeze(dim=0).to(torch.float32)
    # encoder
    # y, kv_catch = test_transformer_encoder(x)
    # decoder
    # result = test_transformer_decoder(y, kv_catch)
    # test_transformer_sep()
    test_transformer_unet(x)

