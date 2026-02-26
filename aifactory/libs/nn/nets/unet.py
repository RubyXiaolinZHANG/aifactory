import torch
from copy import deepcopy
# from dataclasses import dataclass
from aifactory.libs.nn.blocks import BasicBlockParameters, from_dict, ConvSequence
from aifactory.libs.nn.blocks.conv_sequence import build_convolution_sequence
from aifactory.libs.nn.layers import ACTIVATIONS
from aifactory.utils.dict_operator import dict_to_struct_recursive

'''
@dataclass()
class UnetArcParameters:
    adapter: BasicArcParameters
    encoder: list[BasicArcParameters]
    mid_layer: BasicArcParameters
    decoder: list[BasicArcParameters]
    heads: list[BasicArcParameters]


'''


def build_encoder_stage(cls, input_channels, output_channel, activation=None):
    if isinstance(input_channels, int):
        input_channels = [input_channels]
    if activation is None:
        activation = {"name": "relu",
                      "parameters": None}
    elif isinstance(activation, str):
        activation = {"name": activation,
                      "parameters": None}

    if activation.get("parameters", None) is None:
        activation = ACTIVATIONS[activation["name"].lower()]()
    elif isinstance(activation["parameters"], (tuple, list)):
        activation = ACTIVATIONS[activation["name"].lower()](*activation["parameters"])
    elif isinstance(activation["parameters"], dict):
        activation = ACTIVATIONS[activation["name"].lower()](**activation["parameters"])
    else:
        raise ValueError("Do not support activation parameters type: {}".format(type(activation["parameters"])))

    layers = []
    for layer_id, ic in enumerate(input_channels):
        if layer_id == len(input_channels) - 1:
            oc = output_channel
            # conv layer
            layers.append(torch.nn.Conv2d(ic, oc, 3, stride=1, padding=1))
        else:
            oc = input_channels[layer_id + 1]
        setattr(cls,
                "conv_{}".format(layer_id),
                layers[-1])
        # activation layer
        layers.append(deepcopy(activation))
        setattr(cls,
                "act_{}".format(layer_id),
                layers[-1])

    setattr(cls, "_layers", layers)
    return


class EncoderStage(torch.nn.Module):

    def __init__(self, arc_parameters):
        build_unet_encoder(self, arc_parameters)

    def forward(self, x):
        for stage in self._stages:
            x = stage(x)
        return x


def build_unet_encoder(cls, stages_parameters):

    for stage_name, stage_params in stages_parameters.items():
        setattr(cls, stage_name, torch.nn.Module())
        build_encoder_stage(getattr(cls, stage_name), stage_params)


class UnetEncoder(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_encoder_stage(stages_parameters)

    def forward(self, x):
        for stage in self._stages:
            x = stage(x)
        return x

    @property
    def stages(self):
        return self._stages if hasattr(self, "_stages") and self._stages is not None else None


def build_decoder_stage(cls, input_channels, output_channel, activation=None):
    if isinstance(input_channels, int):
        input_channels = [input_channels]
    if activation is None:
        activation = {"name": "relu",
                      "parameters": None}
    elif isinstance(activation, str):
        activation = {"name": activation,
                      "parameters": None}

    if activation.get("parameters", None) is None:
        activation = ACTIVATIONS[activation["name"].lower()]()
    elif isinstance(activation["parameters"], (tuple, list)):
        activation = ACTIVATIONS[activation["name"].lower()](*activation["parameters"])
    elif isinstance(activation["parameters"], dict):
        activation = ACTIVATIONS[activation["name"].lower()](**activation["parameters"])
    else:
        raise ValueError("Do not support activation parameters type: {}".format(type(activation["parameters"])))

    layers = []
    for layer_id, ic in enumerate(input_channels):
        if layer_id == len(input_channels) - 1:
            oc = output_channel
            # conv layer
            layers.append(torch.nn.Conv2d(ic, oc, 3, stride=1, padding=1))
        else:
            oc = input_channels[layer_id + 1]
        setattr(cls,
                "conv_{}".format(layer_id),
                layers[-1])
        # activation layer
        layers.append(deepcopy(activation))
        setattr(cls,
                "act_{}".format(layer_id),
                layers[-1])

    setattr(cls, "_layers", layers)
    return


class DecoderStage(torch.nn.Module):

    def __init__(self, arc_parameters):
        super().__init__()
        build_unet_decoder(self, arc_parameters)

    def forward(self, x):
        for stage in self._stages:
            x = stage(x)
        return x


def build_unet_decoder(cls, stages_parameters):

    for stage_name, stage_params in stages_parameters.items():
        setattr(cls, stage_name, torch.nn.Module())
        build_encoder_stage(getattr(cls, stage_name), stage_params)


class UnetDecoder(torch.nn.Module):

    def __init__(self, stages_parameters):
        super().__init__()
        build_decoder_stage(stages_parameters)

    def forward(self, x):
        for stage in self._stages:
            x = stage(x)
        return x

    @property
    def stages(self):
        return self._stages if hasattr(self, "_stages") and self._stages is not None else None


def build_unet_head(cls, input_channel,  head_parameters):
    for head_name, parameters in head_parameters.items():
        mid_channels = parameters.get("mid_channels", None)
        if mid_channels is None:
            mid_channels = []
        elif isinstance(mid_channels, int):
            mid_channels = [mid_channels]
        elif isinstance(mid_channels, (tuple, list)):
            pass
        else:
            raise ValueError("Do not support Unet {} mid_chanel type: {}".format(head_name, type(mid_channels)))

        setattr(cls, head_name, torch.nn.Module())
        build_convolution_sequence(getattr(cls, head_name),
                                   [input_channel] + [mid_channels],
                                   head_parameters["output_channel"],
                                   activation=head_parameters.get("activation", None))



class UnetHead(torch.nn.Module):

    def __init__(self, input_channel, output_channels):
        super().__init__()


class Unet(torch.nn.Module):

    def __init__(self, unet_arc_parameters):
        super().__init__()
        self._adaptor = ConvSequence(unet_arc_parameters.adapter)
        self._encoder = UnetEncoder(unet_arc_parameters.encoder)
        self._mid_layer = ConvSequence(unet_arc_parameters.mid_layer)
        self._decoder = UnetDecoder(unet_arc_parameters.decoder)
        self._heads = UnetHead(unet_arc_parameters.head)



def test_adaptor():
    from aifactory.utils.load_file import load_file
    yaml_file = r"D:\Program\ToGit\xiaomi\aifactory\model_zoo\ainr\configs\AINR_Unet_Tuning.yaml"
    config = load_file(yaml_file)
    params = config["model"]["parameters"]['arch']['adaptor']
    params = from_dict(BasicBlockParameters, params)
    unet_adaptor = ConvSequence(params)
    print(unet_adaptor)


if __name__ == "__main__":
    test_adaptor()