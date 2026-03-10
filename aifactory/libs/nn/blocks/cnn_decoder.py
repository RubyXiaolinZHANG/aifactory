import torch
from copy import deepcopy
from aifactory.libs.nn.blocks import ConvSequence
from aifactory.libs.nn.layers import get_activation_by_name


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

