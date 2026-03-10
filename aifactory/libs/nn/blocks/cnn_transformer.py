import torch
from dataclasses import dataclass
from aifactory.libs.nn.blocks.arc_parameters import BasicBlockParameters
from aifactory.libs.nn.blocks.cnn_decoder import DecoderStage
from aifactory.libs.nn.blocks.cnn_encoder import EncoderStage
from aifactory.libs.nn.blocks.cnn_encoder import ConvSequence


@dataclass
class CnnTransformerAttentionBlockParameters:
    in_channels: int
    mid_channels: int
    bias: bool = True


@dataclass
class CnnTransformerCrossAttentionBlockParameters:
    in_channels: int
    mid_channels: int
    bias: bool = True


@dataclass
class CnnTransformerMlpBlockParameters:
    in_channels: int
    mid_channels: int
    bias: bool = True


@dataclass
class CnnTransformerInteractBlockParameters:
    in_channels: int | None
    mid_channels: list | tuple | None
    out_channels: int | None
    mode: str
    activation: dict | None
    bias: bool = True
    pre_activation: bool = False
    output_activation_disable: bool = False
    in_front: bool = False
    disable: bool = False


@dataclass
class CnnTransformerUpSampleBlockParameters(BasicBlockParameters):
    in_front: bool = False
    disable: bool = False


class CnnTransformerAttentionBlock(torch.nn.Module):

    def __init__(self, in_channels, mid_channels, bias=True):
        super().__init__()
        self._in_channels = in_channels
        self._mid_channels = mid_channels
        # self._out_channels = out_channels
        self._bias = bias
        self.kqv_projection = torch.nn.Conv2d(in_channels, mid_channels * 3, 3, stride=1, padding=1, bias=bias)
        self.scoring = torch.nn.Sigmoid()
        self.o_projection = torch.nn.Conv2d(mid_channels, in_channels, 3, stride=1, padding=1, bias=bias)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def mid_channels(self):
        return self._mid_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        projections = self.kqv_projection(x)
        q = projections[:, :self._mid_channels, :, :]
        k = projections[:, self._mid_channels:self._mid_channels * 2, :, :]
        v = projections[:, self._mid_channels * 2:, :, :]
        score = self.scoring(q * k)
        val = v * score
        attn_out = self.o_projection(val) + x
        return attn_out, {"k": k,
                          "v": v}


class CnnTransformerCrossAttentionBlock(torch.nn.Module):

    def __init__(self, in_channels, mid_channels, bias=True):
        super().__init__()
        self._in_channels = in_channels
        self._mid_channels = mid_channels
        # self._out_channels = out_channels
        self._bias = bias
        self.q_projection = torch.nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1, bias=bias)
        self.scoring = torch.nn.Sigmoid()
        self.o_projection = torch.nn.Conv2d(mid_channels, in_channels, 3, stride=1, padding=1, bias=bias)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def mid_channels(self):
        return self._mid_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x, k, v):
        q = self.q_projection(x)
        score = self.scoring(q * k)
        val = v * score
        attn_out = self.o_projection(val) + x
        return attn_out


class CnnTransformerMlpBlock(torch.nn.Module):

    def __init__(self, in_channels, mid_channels, bias=True):
        super().__init__()
        self._in_channels = in_channels
        self._mid_channels = mid_channels
        self._bias = bias
        self.g_projection = torch.nn.Conv2d(in_channels, mid_channels * 2, 3, stride=1, padding=1, bias=bias)
        self.act = torch.nn.GELU()
        self.o_projection = torch.nn.Conv2d(mid_channels, in_channels, 3, stride=1, padding=1, bias=bias)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def mid_channels(self):
        return self._mid_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        gating = self.g_projection(x)
        gating = self.act(gating[:, :self._mid_channels, :, :]) * gating[:, self._mid_channels:, :, :]
        mlp_out = self.o_projection(gating) + x
        return mlp_out


class CnnTransformerInteractBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 mode,
                 activation=None,
                 bias=True,
                 pre_activation=False,
                 output_activation_disable=False,
                 disable=False):
        super().__init__()
        self._disable = disable
        if self._disable:
            return
        elif mode == "down_sample":
            setattr(self, "down_sample", EncoderStage(in_channels,
                                                      mid_channels,
                                                      out_channels,
                                                      activation,
                                                      bias,
                                                      pre_activation,
                                                      output_activation_disable))
            setattr(self, "_implement", getattr(self, "down_sample"))
        elif mode == "up_sample":
            setattr(self, "up_sample", DecoderStage(in_channels,
                                                    mid_channels,
                                                    out_channels,
                                                    activation,
                                                    bias,
                                                    pre_activation,
                                                    output_activation_disable))
            setattr(self, "_implement", getattr(self, "up_sample"))
        elif mode == "keep_resolution":
            setattr(self, "projection", ConvSequence(in_channels,
                                                     mid_channels,
                                                     out_channels,
                                                     activation,
                                                     bias,
                                                     pre_activation,
                                                     output_activation_disable))
            setattr(self, "_implement", getattr(self, "projection"))
        else:
            raise ValueError(
                "unrecognized mode: {}. Should be \'down_sample\', \'up_sample\' or \'projection\' ".format(mode))

    def forward(self, x):
        if self._disable:
            return x
        else:
            return self._implement(x)


class CnnTransformerEncoderBlock(torch.nn.Module):

    def __init__(self, attn_block_params: CnnTransformerAttentionBlockParameters,
                 mlp_params: CnnTransformerMlpBlockParameters,
                 down_sample_params: CnnTransformerInteractBlockParameters | None):
        super().__init__()
        # init down sample block
        if down_sample_params is None or down_sample_params.disable:
            self._down_sample_in_front = False
        else:
            if down_sample_params.disable:
                pass
            elif down_sample_params.mode.lower() == "down_sample" or down_sample_params.mode.lower() == "down sample":
                self.down_sample_norm = torch.nn.InstanceNorm2d(down_sample_params.in_channels)
                self.down_sample = EncoderStage(down_sample_params.in_channels,
                                                down_sample_params.mid_channels,
                                                down_sample_params.out_channels,
                                                down_sample_params.activation,
                                                down_sample_params.bias,
                                                down_sample_params.pre_activation,
                                                down_sample_params.output_activation_disable)
            elif down_sample_params.mode == "keep_resolution" or down_sample_params.mode.lower() == "keep resolution":
                self.projection_norm = torch.nn.InstanceNorm2d(down_sample_params.in_channels)
                self.projection = ConvSequence(down_sample_params.in_channels,
                                               down_sample_params.mid_channels,
                                               down_sample_params.out_channels,
                                               down_sample_params.activation,
                                               down_sample_params.bias,
                                               down_sample_params.pre_activation,
                                               down_sample_params.output_activation_disable)
            self._down_sample_in_front = down_sample_params.in_front
        # init input normalizer
        if self._down_sample_in_front:
            self.pre_attn_norm = torch.nn.InstanceNorm2d(down_sample_params.in_channels)
        else:
            self.pre_attn_norm = torch.nn.InstanceNorm2d(attn_block_params.in_channels)
        # init attention block
        self.attn = CnnTransformerAttentionBlock(attn_block_params.in_channels,
                                                 attn_block_params.mid_channels,
                                                 bias=attn_block_params.bias)
        # init post attention normalizer
        self.post_attn_norm = torch.nn.InstanceNorm2d(mlp_params.in_channels)
        # init MLP
        self.mlp = CnnTransformerMlpBlock(mlp_params.in_channels,
                                          mlp_params.mid_channels,
                                          bias=mlp_params.bias)

    def forward(self, x):
        if self._down_sample_in_front:
            if hasattr(self, "down_sample"):
                x = self.down_sample(self.down_sample_norm(x))
            elif hasattr(self, "projection"):
                x = self.projection(x)
            else:
                pass
        x_norm = self.pre_attn_norm(x)
        attn, kv = self.attn(x_norm)
        attn_norm = self.post_attn_norm(attn)
        y = self.mlp(attn_norm)
        if not self._down_sample_in_front:
            if hasattr(self, "down_sample"):
                y = self.down_sample(self.down_sample_norm(y))
            elif hasattr(self, "projection"):
                y = self.projection(self.projection_norm(y))
            else:
                pass
        return y, kv


class CnnTransformerDecoderBlock(torch.nn.Module):

    def __init__(self, self_attn_block_params: CnnTransformerAttentionBlockParameters | None,
                 cross_attn_block_params: CnnTransformerAttentionBlockParameters | None,
                 mlp_params: CnnTransformerMlpBlockParameters,
                 up_sample_params: CnnTransformerInteractBlockParameters | None):
        super().__init__()
        # init down sample block
        if up_sample_params is None or up_sample_params.disable:
            self._up_sample_in_front = False
        else:
            if up_sample_params.disable:
                pass
            elif up_sample_params.mode.lower() == "up_sample" or up_sample_params.mode.lower() == "up sample":
                self.up_sample_norm = torch.nn.InstanceNorm2d(up_sample_params.in_channels)
                self.up_sample = DecoderStage(up_sample_params.in_channels,
                                              up_sample_params.mid_channels,
                                              up_sample_params.out_channels,
                                              up_sample_params.activation,
                                              up_sample_params.bias,
                                              up_sample_params.pre_activation,
                                              up_sample_params.output_activation_disable)
            elif up_sample_params.mode.lower() == "keep_resolution" or up_sample_params.mode.lower() == "keep resolution":
                self.projection_norm = torch.nn.InstanceNorm2d(up_sample_params.in_channels)
                self.projection = ConvSequence(up_sample_params.in_channels,
                                               up_sample_params.mid_channels,
                                               up_sample_params.out_channels,
                                               up_sample_params.activation,
                                               up_sample_params.bias,
                                               up_sample_params.pre_activation,
                                               up_sample_params.output_activation_disable)
            self._up_sample_in_front = up_sample_params.in_front
        # init input normalizer
        if self._up_sample_in_front:
            self.pre_attn_norm = torch.nn.InstanceNorm2d(up_sample_params.in_channels)
        else:
            self.pre_attn_norm = torch.nn.InstanceNorm2d(mlp_params.in_channels)
        # init self attention block
        if self_attn_block_params is None:
            self.self_attn = None
        else:
            self.self_attn = CnnTransformerAttentionBlock(self_attn_block_params.in_channels,
                                                          self_attn_block_params.mid_channels,
                                                          bias=self_attn_block_params.bias)
        # init cross attention block
        if cross_attn_block_params is None:
            self.cross_attn = None
        else:
            self.cross_attn = CnnTransformerCrossAttentionBlock(cross_attn_block_params.in_channels,
                                                                cross_attn_block_params.mid_channels,
                                                                bias=cross_attn_block_params.bias)
        # init post attention normalizer
        self.post_attn_norm = torch.nn.InstanceNorm2d(mlp_params.in_channels)
        self.post_cross_attn_norm = torch.nn.InstanceNorm2d(mlp_params.in_channels)
        # init MLP
        self.mlp = CnnTransformerMlpBlock(mlp_params.in_channels,
                                          mlp_params.mid_channels,
                                          bias=mlp_params.bias)

    def forward(self, x, k, v):
        if self._up_sample_in_front:
            if hasattr(self, "up_sample"):
                x = self.up_sample(self.up_sample_norm(x))
            elif hasattr(self, "projection"):
                x = self.projection(self.projection_norm(x))
            else:
                pass
        # self attention
        x_norm = self.pre_attn_norm(x)
        if self.self_attn is None:
            attn_norm = x_norm
        else:
            try:
                attn, _= self.self_attn(x_norm)
                attn_norm = self.post_attn_norm(attn)
            except:
                print("**************************")
        # cross attention
        if self.cross_attn is not None:
            attn = self.cross_attn(attn_norm, k, v)
            attn_norm = self.post_cross_attn_norm(attn)
        # feed forward
        y = self.mlp(attn_norm)
        if not self._up_sample_in_front:
            if hasattr(self, "up_sample"):
                y = self.up_sample(self.up_sample_norm(y))
            elif hasattr(self, "projection"):
                y = self.projection(self.projection_norm(y))
            else:
                pass
        return y


######################################################################################
# the followings are test codes
######################################################################################


def test_attn_block(x):
    from torchinfo import summary
    attn = CnnTransformerAttentionBlock(3, 16)
    y, kv = attn(x)
    summary(attn, input_data=x)
    print("x: {}".format(x.shape))
    print("y: {}".format(y.shape))


def test_mlp_block(x):
    from torchinfo import summary
    mlp = CnnTransformerMlpBlock(3, 16)
    y = mlp(x)
    summary(mlp, input_data=x)
    print("x: {}".format(x.shape))
    print("y: {}".format(y.shape))


def test_decoder_block(x):
    from torchinfo import summary
    decoder = EncoderStage(3,
                           None,
                           6,
                           activation="relu",
                           bias=True)
    y = decoder(x)
    summary(decoder, input_data=x)
    print("x: {}".format(x.shape))
    print("y: {}".format(y.shape))


def test_transformer_encoder(x):
    import os, onnx
    from torchinfo import summary
    from aifactory.libs.nn.blocks import from_dict
    from aifactory.utils.load_file import load_file
    yaml_file = r"D:\Program\ToGit\xiaomi\aifactory\model_zoo\ainr\configs\models\cnn_transformer_encoder.yaml"
    onnx_file = os.path.join("../../../../onnx/", "cnn_transformer_encoder.onnx")
    config = load_file(yaml_file)

    arc_params = config["parameters"]["arch"]
    arc_params['attn_block_params'] = from_dict(CnnTransformerAttentionBlockParameters, arc_params['attn_block_params'])
    arc_params['mlp_params'] = from_dict(CnnTransformerMlpBlockParameters, arc_params['mlp_params'])
    arc_params['down_sample_params'] = from_dict(CnnTransformerInteractBlockParameters,
                                                 arc_params['down_sample_params'])

    transformer_encoder = CnnTransformerEncoderBlock(arc_params['attn_block_params'],
                                                     arc_params['mlp_params'],
                                                     arc_params['down_sample_params'])
    transformer_encoder.eval()
    '''
    for module in transformer_decoder.modules():
        if isinstance(module, (torch.nn.InstanceNorm2d, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm3d)):
            module.train(False)
            module.track_running_stats = True
    '''

    y, kv = transformer_encoder(x)
    summary(transformer_encoder, input_data=x)
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(transformer_encoder, x, onnx_file,
                      opset_version=11,
                      training=torch.onnx.TrainingMode.EVAL)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)
    print("x: {}".format(x.shape))
    print("y: {}, min={}, max={}".format(y.shape, y.min(), y.max()))
    print("save onnx to: {}".format(os.path.abspath(onnx_file)))
    return y, kv


def test_transformer_decoder(x, kv):
    import os, onnx
    from torchinfo import summary
    from aifactory.libs.nn.blocks import from_dict
    from aifactory.utils.load_file import load_file
    yaml_file = r"D:\Program\ToGit\xiaomi\aifactory\model_zoo\ainr\configs\models\cnn_transformer_decoder.yaml"
    onnx_file = os.path.join("../../../../onnx/", "cnn_transformer_decoder.onnx")
    config = load_file(yaml_file)

    arc_params = config["parameters"]["arch"]
    arc_params['self_attn_block_params'] = from_dict(CnnTransformerAttentionBlockParameters,
                                                      arc_params['cross_attn_block_params'])
    arc_params['cross_attn_block_params'] = from_dict(CnnTransformerAttentionBlockParameters,
                                                      arc_params['cross_attn_block_params'])
    arc_params['mlp_params'] = from_dict(CnnTransformerMlpBlockParameters, arc_params['mlp_params'])
    arc_params['up_sample_params'] = from_dict(CnnTransformerInteractBlockParameters,
                                               arc_params['up_sample_params'])

    transformer_decoder = CnnTransformerDecoderBlock(arc_params['self_attn_block_params'],
                                                     arc_params['cross_attn_block_params'],
                                                     arc_params['mlp_params'],
                                                     arc_params['up_sample_params'])
    transformer_decoder.eval()
    print(transformer_decoder)

    y = transformer_decoder(x, kv["k"], kv['v'])
    summary(transformer_decoder, input_data=(x, kv["k"], kv['v']))
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(transformer_decoder, (x, kv["k"], kv['v']), onnx_file,
                      opset_version=11,
                      training=torch.onnx.TrainingMode.EVAL)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)
    print("x: {}".format(x.shape))
    print("y: {}, min={}, max={}, mean={}, std={}".format(y.shape, y.min(), y.max(), y.mean(), y.std()))
    print("save onnx to: {}".format(os.path.abspath(onnx_file)))
    return y


if __name__ == "__main__":
    import cv2

    image = cv2.imread("../../../../tests/images/llama.jpg")
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0).to(torch.float32)

    # test_attn_block(x)
    # test_mlp_block(x)
    # test_decoder_block(x)
    y, kv = test_transformer_encoder(x)

    # y = torch.randn([1, 16, 438, 657])
    x_hat = test_transformer_decoder(y, kv)
    # cv2.imshow("lamma", image)
    # cv2.waitKey()
    # attn = CnnTransformerAttentionBlock(3, 16, 16)
    # y = attn(x)
