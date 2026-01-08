from torch import nn

from bitnet.nn import BitConv2d, BitLinear


def replace_linear_layers(model: nn.Module) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                BitLinear(module.in_features, module.out_features, bias=module.bias is not None),
            )
        else:
            replace_linear_layers(module)


def replace_conv2d_layers(model: nn.Module) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(
                model,
                name,
                BitConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    bias=module.bias is not None,
                ),
            )
        else:
            replace_conv2d_layers(module)


def replace_layers(model: nn.Module) -> None:
    replace_linear_layers(model)
    replace_conv2d_layers(model)
