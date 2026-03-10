import os
import torch
from copy import deepcopy
from aifactory.libs.nn.blocks import BasicBlockParameters, from_dict, ConvSequence, ConvSequenceBlock
from aifactory.libs.nn.layers import get_activation_by_name


# encoder block
def build_encoder_stage(cls, in_channels, mid_channels, out_channels, activation=None, bias=True,
                        pre_activation=False, output_activation_disable=False):
    # all conv input channels
    if mid_channels is None:
        mid_channels = []
    elif isinstance(mid_channels, tuple):
        mid_channels = list(mid_channels)
    in_channels = [in_channels] + mid_channels

    # init activation
    if activation is None:
        pass
    elif isinstance(activation, str):
        activation = {"name": activation,
                      "parameters": None}

    if activation is not None:
        activation = get_activation_by_name(activation["name"].lower(), activation.get("parameters"))

    layers = []
    for layer_id, ic in enumerate(in_channels):
        # place activation before conv
        if pre_activation and activation is not None:
            layers.append(deepcopy(activation))
            setattr(cls,
                    "act_{}".format(layer_id),
                    layers[-1])

        # conv layer
        if layer_id == len(in_channels) - 1:
            oc = out_channels
            by_pass_activation = output_activation_disable
        else:
            oc = in_channels[layer_id + 1]
            by_pass_activation = False
        if layer_id == 0:
            layers.append(torch.nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=bias))
        else:
            layers.append(torch.nn.Conv2d(ic, oc, 3, stride=1, padding=1, bias=bias))
        setattr(cls,
                "conv_{}".format(layer_id),
                layers[-1])

        # # place activation before conv
        if by_pass_activation or pre_activation:
            continue
        if activation is not None:
            layers.append(deepcopy(activation))
            setattr(cls,
                    "act_{}".format(layer_id),
                    layers[-1])

    setattr(cls, "_layers", layers)
    return


class EncoderStage(ConvSequence):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 activation,
                 bias,
                 pre_activation,
                 output_activation_disable):
        torch.nn.Module.__init__(self)
        build_encoder_stage(self,
                            in_channels,
                            mid_channels,
                            out_channels,
                            activation,
                            bias,
                            pre_activation,
                            output_activation_disable)


# unet encoder
def build_unet_encoder(cls, stages_parameters):
    stages = []
    for stage_name, stage_params in stages_parameters.items():
        setattr(cls, stage_name, EncoderStage(stage_params.in_channels,
                                              stage_params.mid_channels,
                                              stage_params.out_channels,
                                              stage_params.activation,
                                              stage_params.bias,
                                              stage_params.pre_activation,
                                              stage_params.output_activation_disable))
        stages.append(getattr(cls, stage_name))
    setattr(cls, "_stages", stages)
    return


class UnetEncoder(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_unet_encoder(self, stages_parameters)

    def forward(self, x):
        assert self.stage_num > 0
        skip_connections = {}
        for stage_id, stage in enumerate(self._stages):
            x = stage(x)
            if stage_id < self.stage_num - 1:
                skip_connections["stage_{}".format(stage_id)] = x
        return x, skip_connections

    @property
    def stages(self):
        return self._stages if hasattr(self, "_stages") and self._stages is not None else None

    @property
    def stage_num(self):
        return 0 if self.stages is None else len(self.stages)


# decoder block
def build_decoder_stage(cls, in_channels, mid_channels, out_channels, activation=None, bias=True,
                        pre_activation=False, output_activation_disable=False):
    # all conv input channels
    if mid_channels is None:
        mid_channels = []
    elif isinstance(mid_channels, tuple):
        mid_channels = list(mid_channels)
    in_channels = [in_channels] + mid_channels

    if activation is None:
        pass
    elif isinstance(activation, str):
        activation = {"name": activation,
                      "parameters": None}
    if activation is not None:
        activation = get_activation_by_name(activation["name"].lower(), activation.get("parameters"))

    layers = []
    for layer_id, ic in enumerate(in_channels):
        # place activation before conv
        if pre_activation and activation is not None:
            layers.append(deepcopy(activation))
            setattr(cls,
                    "act_{}".format(layer_id),
                    layers[-1])
        # conv layer
        if layer_id == len(in_channels) - 1:
            oc = out_channels
            # transpose conv layer
            layers.append(torch.nn.ConvTranspose2d(ic, oc, 2, stride=2, padding=0, bias=bias))
            by_pass_activation = output_activation_disable
        else:
            oc = in_channels[layer_id + 1]
            # conv layer
            layers.append(torch.nn.Conv2d(ic, oc, 3, stride=1, padding=1, bias=bias))
            by_pass_activation = False

        setattr(cls,
                "conv_{}".format(layer_id),
                layers[-1])
        # place activation after conv
        if by_pass_activation or pre_activation:
            continue

        if activation is not None:
            layers.append(deepcopy(activation))
            setattr(cls,
                    "act_{}".format(layer_id),
                    layers[-1])

    setattr(cls, "_layers", layers)
    return


class DecoderStage(ConvSequence):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 activation,
                 bias,
                 pre_activation,
                 output_activation_disable):
        torch.nn.Module.__init__(self)
        build_decoder_stage(self,
                            in_channels,
                            mid_channels,
                            out_channels,
                            activation,
                            bias,
                            pre_activation,
                            output_activation_disable)


# unet decoder
def build_unet_decoder(cls, stages_parameters):
    stages = []
    for stage_name, stage_params in stages_parameters.items():
        setattr(cls, stage_name, DecoderStage(stage_params.in_channels,
                                              stage_params.mid_channels,
                                              stage_params.out_channels,
                                              stage_params.activation,
                                              stage_params.bias,
                                              stage_params.pre_activation,
                                              stage_params.output_activation_disable))
        stages.append(getattr(cls, stage_name))
    setattr(cls, "_stages", stages)


class UnetDecoder(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_unet_decoder(self, stages_parameters)

    def forward(self, x, skip_connections):
        if isinstance(skip_connections, dict):
            skip_connections = list(skip_connections.values())
        elif isinstance(skip_connections, tuple):
            skip_connections = list(skip_connections)
        elif isinstance(skip_connections, list):
            pass
        else:
            raise ValueError("Only support skip connection type of dict, tuple list, but this is {}".format(
                type(skip_connections)))
        skip_connections.reverse()
        for stage_id, stage in enumerate(self._stages):
            if stage_id == 0:
                x = stage(x)
            else:
                x = stage(torch.concat([x, skip_connections[stage_id - 1]], dim=1))
        return x

    @property
    def stages(self):
        return self._stages if hasattr(self, "_stages") and self._stages is not None else None


# unet heads
def build_unet_head(cls, head_parameter_dict):
    heads = []
    for head_name, head_parameters in head_parameter_dict.items():
        setattr(cls, head_name, ConvSequence(head_parameters.in_channels,
                                             head_parameters.mid_channels,
                                             head_parameters.out_channels,
                                             activation=head_parameters.activation,
                                             bias=head_parameters.bias,
                                             pre_activation=head_parameters.pre_activation,
                                             output_activation_disable=head_parameters.output_activation_disable))
        heads.append(getattr(cls, head_name))
    setattr(cls, "_heads", heads)
    return


class UnetHead(torch.nn.Module):

    def __init__(self, head_params):
        super().__init__()
        build_unet_head(self, head_params)

    def forward(self, x):
        y = []
        for head in self._heads:
            y.append(head(x))
        return y

    @property
    def heads(self):
        return self._heads if hasattr(self, "_heads") and self._heads is not None else None


# unet
class Unet(torch.nn.Module):

    def __init__(self, unet_arc_parameters):
        super().__init__()
        self.adaptor = ConvSequence(unet_arc_parameters["adaptor"].in_channels,
                                    unet_arc_parameters["adaptor"].mid_channels,
                                    unet_arc_parameters["adaptor"].out_channels,
                                    activation=unet_arc_parameters["adaptor"].activation,
                                    bias=unet_arc_parameters["adaptor"].bias,
                                    pre_activation=unet_arc_parameters["adaptor"].pre_activation,
                                    output_activation_disable=unet_arc_parameters["adaptor"].output_activation_disable)
        self.encoder = UnetEncoder(unet_arc_parameters['encoder'])
        if unet_arc_parameters.get("mid_layer") is None:
            self.mid_layer = None
        elif isinstance(unet_arc_parameters["mid_layer"], dict):
            self.mid_layer = ConvSequenceBlock(unet_arc_parameters['mid_layer'])
        elif isinstance(unet_arc_parameters["mid_layer"], BasicBlockParameters):
            self.mid_layer = ConvSequence(unet_arc_parameters['mid_layer'].in_channels,
                                          unet_arc_parameters["mid_layer"].mid_channels,
                                          unet_arc_parameters["mid_layer"].out_channels,
                                          activation=unet_arc_parameters["mid_layer"].activation,
                                          bias=unet_arc_parameters["mid_layer"].bias,
                                          pre_activation=unet_arc_parameters["mid_layer"].pre_activation,
                                          output_activation_disable=unet_arc_parameters[
                                              "mid_layer"].output_activation_disable)
        self.decoder = UnetDecoder(unet_arc_parameters['decoder'])
        self.heads = UnetHead(unet_arc_parameters['heads'])

    def forward(self, x):
        latent = self.adaptor(x)
        encoder, skip_connection = self.encoder(latent)
        if self.mid_layer is None:
            mid_latent = encoder
        else:
            mid_latent = self.mid_layer(encoder)
        decoder = self.decoder(mid_latent, skip_connection)
        outputs = self.heads(torch.concat([latent, decoder], dim=1))
        return outputs


######################################################################################
# the followings are test codes
######################################################################################

def test_adaptor(config):
    params = config["model"]["parameters"]['arch']['adaptor']
    params = from_dict(BasicBlockParameters, params)
    unet_adaptor = ConvSequence(params)
    print(unet_adaptor)


def test_encoder(config):
    params = config["model"]["parameters"]['arch']['encoder']
    for stage_name, stage_params in params.items():
        params[stage_name] = from_dict(BasicBlockParameters, stage_params)
    unet_encoder = UnetEncoder(params)
    print(unet_encoder)


def test_decoder(config):
    params = config["model"]["parameters"]['arch']['decoder']
    for stage_name, stage_params in params.items():
        params[stage_name] = from_dict(BasicBlockParameters, stage_params)
    unet_decoder = UnetDecoder(params)
    print(unet_decoder)


def test_head(config):
    params = config["model"]["parameters"]['arch']['heads']
    for head_name, head_params in params.items():
        params[head_name] = from_dict(BasicBlockParameters, head_params)
    unet_head = UnetHead(params)
    print(unet_head)


def test_unet(config, onnx_file=None):
    import os, onnx
    from torchinfo import summary

    # get unet arch parameters
    params = config["model"]["parameters"]['arch']['adaptor']
    adaptor_params = from_dict(BasicBlockParameters, params)

    encoder_params = config["model"]["parameters"]['arch']['encoder']
    for stage_name, stage_params in encoder_params.items():
        encoder_params[stage_name] = from_dict(BasicBlockParameters, stage_params)

    mid_params = config["model"]["parameters"]['arch']['mid_layer']
    if mid_params is None:
        pass
    elif "in_channels" in mid_params:
        mid_params = from_dict(BasicBlockParameters, mid_params)
    else:
        for block_name, block_params in mid_params.items():
            mid_params[block_name] = from_dict(BasicBlockParameters, block_params)

    decoder_params = config["model"]["parameters"]['arch']['decoder']
    for stage_name, stage_params in decoder_params.items():
        decoder_params[stage_name] = from_dict(BasicBlockParameters, stage_params)

    heads_params = config["model"]["parameters"]['arch']['heads']
    for head_name, head_params in heads_params.items():
        heads_params[head_name] = from_dict(BasicBlockParameters, head_params)

    unet_arc_parameters = {"adaptor": adaptor_params,
                           "encoder": encoder_params,
                           "mid_layer": mid_params,
                           "decoder": decoder_params,
                           "heads": heads_params}

    # build unet
    unet = Unet(unet_arc_parameters)

    # random value infer for testing
    x = torch.randn(1, adaptor_params.in_channels, 1504, 2000)
    y = unet(x)
    summary(unet,
            input_data=x)
    # save onnx to root
    if onnx_file is None:
        onnx_file = '../../../../onnx/unet.onnx'
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    torch.onnx.export(unet, x, onnx_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(onnx_file)), onnx_file)

    # print info
    print("y: {}".format(y[0].shape))
    print("save onnx to: {}".format(os.path.abspath(onnx_file)))


if __name__ == "__main__":
    from aifactory.utils.load_file import load_file

    # yaml_file = r"D:\Program\ToGit\xiaomi\aifactory\model_zoo\ainr\configs\ainr_unet_tuning_baseline.yaml"
    # onnx_file = os.path.join("../../../../onnx/", "ainr_unet_baseline.onnx")
    yaml_file = r"D:\Program\ToGit\xiaomi\aifactory\model_zoo\ainr\configs\ainr_unet_tuning_transConv_wo_relu.yaml"
    onnx_file = os.path.join("../../../../onnx/", "ainr_unet_transConv_wo_relu.onnx")
    config = load_file(yaml_file)

    # test_adaptor(config) # pass
    # test_encoder(config)  # pass
    # test_decoder(config)
    # test_head(config)
    test_unet(config, onnx_file)
