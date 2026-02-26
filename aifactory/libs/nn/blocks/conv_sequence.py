import torch
from copy import deepcopy
from aifactory.libs.nn.layers import ACTIVATIONS
from aifactory.libs.nn.blocks import BasicBlockParameters


def build_convolution_sequence(cls, input_channel, mid_channels, output_channel, activation=None):
    if mid_channels is  None:
        mid_channels = []
    elif isinstance(mid_channels, tuple):
        mid_channels = list(mid_channels)
    input_channels = [input_channel] + mid_channels
    if activation is None:
        activation = {"name": "relu",
                      "parameters": None}
    elif isinstance(activation, str):
        activation = {"name": activation,
                      "parameters": None}

    if activation.get("parameters", None)  is None:
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
        else:
            oc = input_channels[layer_id + 1]
        # conv layer
        layers.append(torch.nn.Conv2d(ic, oc, 3, stride=1, padding=1))
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


class ConvSequence(torch.nn.Module):

    def __init__(self, arc_param: BasicBlockParameters):
        super().__init__()
        build_convolution_sequence(self,
                                   input_channel=arc_param.input_channel,
                                   mid_channels=arc_param.mid_channels,
                                   output_channel=arc_param.output_channel,
                                   activation=arc_param.activation)

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    @property
    def layers(self):
        return self._layers if hasattr(self, "layers") and self._layers is not None else None
