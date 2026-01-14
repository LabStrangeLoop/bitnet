"""Tests for selective layer replacement (layer-wise ablation)."""

import timm
import torch
from torch import nn

from bitnet.layer_swap_selective import replace_layers_selective
from bitnet.nn import BitConv2d, BitLinear


def count_layer_types(model: nn.Module) -> dict[str, int]:
    """Count occurrences of each layer type in a model."""
    counts: dict[str, int] = {"Conv2d": 0, "BitConv2d": 0, "Linear": 0, "BitLinear": 0}
    for module in model.modules():
        if isinstance(module, BitConv2d):
            counts["BitConv2d"] += 1
        elif isinstance(module, nn.Conv2d):
            counts["Conv2d"] += 1
        elif isinstance(module, BitLinear):
            counts["BitLinear"] += 1
        elif isinstance(module, nn.Linear):
            counts["Linear"] += 1
    return counts


class TestSelectiveReplacement:
    """Tests for replace_layers_selective function."""

    def test_empty_skip_replaces_all(self) -> None:
        """Empty skip_prefixes should replace all layers."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        counts_before = count_layer_types(model)
        assert counts_before["Conv2d"] > 0
        assert counts_before["Linear"] > 0

        replace_layers_selective(model, set())

        counts_after = count_layer_types(model)
        assert counts_after["Conv2d"] == 0
        assert counts_after["Linear"] == 0
        assert counts_after["BitConv2d"] == counts_before["Conv2d"]
        assert counts_after["BitLinear"] == counts_before["Linear"]

    def test_skip_conv1_keeps_first_conv(self) -> None:
        """Skipping 'conv1' should keep the first conv layer in FP32."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"conv1"})

        # conv1 should still be nn.Conv2d
        assert isinstance(model.conv1, nn.Conv2d)
        assert not isinstance(model.conv1, BitConv2d)

        # Other conv layers should be BitConv2d
        assert isinstance(model.layer1[0].conv1, BitConv2d)

    def test_skip_fc_keeps_classifier(self) -> None:
        """Skipping 'fc' should keep the classifier in FP32."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"fc"})

        # fc should still be nn.Linear
        assert isinstance(model.fc, nn.Linear)
        assert not isinstance(model.fc, BitLinear)

        # Conv layers should be BitConv2d
        assert isinstance(model.conv1, BitConv2d)

    def test_skip_layer1_keeps_block(self) -> None:
        """Skipping 'layer1' should keep all layers in that block FP32."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"layer1"})

        # layer1 convs should still be nn.Conv2d
        assert isinstance(model.layer1[0].conv1, nn.Conv2d)
        assert isinstance(model.layer1[0].conv2, nn.Conv2d)

        # layer2+ convs should be BitConv2d
        assert isinstance(model.layer2[0].conv1, BitConv2d)

        # conv1 (stem) should also be BitConv2d
        assert isinstance(model.conv1, BitConv2d)

    def test_skip_layer4_keeps_last_block(self) -> None:
        """Skipping 'layer4' should keep all layers in that block FP32."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"layer4"})

        # layer4 convs should still be nn.Conv2d
        assert isinstance(model.layer4[0].conv1, nn.Conv2d)
        assert isinstance(model.layer4[1].conv2, nn.Conv2d)

        # layer3 convs should be BitConv2d
        assert isinstance(model.layer3[0].conv1, BitConv2d)

    def test_multiple_skips(self) -> None:
        """Can skip multiple layer prefixes at once."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"conv1", "fc"})

        # Both should be FP32
        assert isinstance(model.conv1, nn.Conv2d)
        assert isinstance(model.fc, nn.Linear)

        # Middle layers should be quantized
        assert isinstance(model.layer2[0].conv1, BitConv2d)

    def test_forward_pass_works(self) -> None:
        """Model should still work after selective replacement."""
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"conv1", "fc"})

        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_resnet50_support(self) -> None:
        """Should work with ResNet50 architecture."""
        model = timm.create_model("resnet50", num_classes=10, pretrained=False)
        replace_layers_selective(model, {"layer4"})

        # layer4 has Bottleneck blocks with conv1, conv2, conv3
        assert isinstance(model.layer4[0].conv1, nn.Conv2d)
        assert isinstance(model.layer4[0].conv2, nn.Conv2d)
        assert isinstance(model.layer4[0].conv3, nn.Conv2d)

        # layer3 should be quantized
        assert isinstance(model.layer3[0].conv1, BitConv2d)
