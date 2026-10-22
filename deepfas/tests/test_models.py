import torch
import torch.nn as nn

from src.models import BinaryMapTransformer, LoRAConv2d, ModifiedDeepLabV3_DeepLoRA, count_params


def test_lora_starts_as_original_conv():
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=2, dilation=2)
    x = torch.randn(2, 8, 12, 12)
    expected = conv(x)

    lora = LoRAConv2d(conv, r=4, alpha=8)
    assert torch.allclose(lora(x), expected)                  # lora_B is zero-initialized
    assert not conv.weight.requires_grad and not conv.bias.requires_grad
    assert lora.scaling == 2.0


def test_deeplab_lora_structure():
    model = ModifiedDeepLabV3_DeepLoRA(r=32, alpha=64, pretrained=False)

    lora_layers = [m for m in model.modules() if isinstance(m, LoRAConv2d)]
    assert len(lora_layers) == 3                               # conv2 of the 3 bottlenecks of layer4
    assert model.model.aux_classifier is None
    assert model.model.classifier[4].out_channels == 1

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(("lora_" in n) or n.startswith("model.classifier.4.") for n in trainable)

    # 3 x (512*32*3*3 + 32*512) LoRA weights + (256 + 1) output convolution
    assert count_params(model)[1] == 3 * (512 * 32 * 9 + 32 * 512) + 257

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


def test_transformer_classifier_shapes():
    model = BinaryMapTransformer(image_size=520, patch_size=20)
    assert model.num_patches == 26 * 26
    logits = model(torch.randint(0, 2, (2, 1, 520, 520)).float())
    assert logits.shape == (2, 2)
