import torch
from copy import deepcopy
from aifactory.libs.nn.layers import ACTIVATIONS, get_activation_by_name
from aifactory.libs.nn.blocks import BasicBlockParameters


def build_convolution_sequence(cls, in_channels, mid_channels, out_channels,
                               activation=None, bias=True, pre_activation=False, output_activation_disable=False):
    assert in_channels is not None or activation is not None
    # get activation func
    if activation is None:
        pass
    elif isinstance(activation, str):
        activation = {"name": activation,
                      "parameters": None}
    '''
    if activation is None:
        pass
    elif activation.get("parameters") is None:
        activation = ACTIVATIONS[activation["name"].lower()]()
    elif isinstance(activation["parameters"], (tuple, list)):
        activation = ACTIVATIONS[activation["name"].lower()](*activation["parameters"])
    elif isinstance(activation["parameters"], dict):
        activation = ACTIVATIONS[activation["name"].lower()](**activation["parameters"])
    else:
        raise ValueError("Do not support activation parameters type: {}".format(type(activation["parameters"])))
    
    '''
    if activation is not None:
        activation = get_activation_by_name(activation["name"].lower(), activation.get("parameters"))
    # activation only, no conv
    layers = []
    if in_channels is None:
        layers.append(activation)
        setattr(cls,
                "act",
                layers[-1])
    else:
        if mid_channels is None:
            mid_channels = []
        elif isinstance(mid_channels, tuple):
            mid_channels = list(mid_channels)
        in_channels = [in_channels] + mid_channels

        for layer_id, ic in enumerate(in_channels):

            # place activation before conv
            if pre_activation and activation is not None:
                layers.append(deepcopy(activation))
                setattr(cls,
                        "act_{}".format(layer_id),
                        layers[-1])

            if layer_id == len(in_channels) - 1:
                oc = out_channels
                by_pass_activation = output_activation_disable
            else:
                oc = in_channels[layer_id + 1]
                by_pass_activation = False

            # conv layer
            layers.append(torch.nn.Conv2d(ic, oc, 3, stride=1, padding=1, bias=bias))
            setattr(cls,
                    "conv_{}".format(layer_id),
                    layers[-1])

            # activation layer
            if by_pass_activation or pre_activation:
                continue

            # place activation after conv
            if activation is not None:
                layers.append(deepcopy(activation))
                setattr(cls,
                        "act_{}".format(layer_id),
                        layers[-1])

    setattr(cls, "_layers", layers)
    return


class ConvSequence(torch.nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels,
                 activation=None, bias=True, pre_activation=False, output_activation_disable=False):
        super().__init__()
        build_convolution_sequence(self,
                                   in_channels,
                                   mid_channels,
                                   out_channels,
                                   activation=activation,
                                   bias=bias,
                                   pre_activation=pre_activation,
                                   output_activation_disable=output_activation_disable)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    # @property
    # def layers(self):
    #     return self._layers if hasattr(self, "layers") and self._layers is not None else None


class ConvSequenceBlock(torch.nn.Module):

    def __init__(self, arc_param):
        super().__init__()

        blocks = []
        for block_name, block_params in arc_param.items():
            conv_seq = ConvSequence(block_params.in_channels,
                                    block_params.mid_channels,
                                    block_params.out_channels,
                                    activation=block_params.activation,
                                    bias=block_params.bias,
                                    pre_activation=block_params.pre_activation,
                                    output_activation_disable=block_params.output_activation_disable)
            blocks.append(conv_seq)
            setattr(self, block_name, conv_seq)
        setattr(self, "_blocks", blocks)

    def forward(self, x):
        for block in self._blocks:
            x = block(x)
        return x
