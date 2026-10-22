"""
Models of the pipeline.

* ``get_mask_creator_model``     - off-the-shelf DeepLabV3 used to create the
                                   training masks of live samples.
* ``LoRAConv2d``                 - LoRA wrapper for ``nn.Conv2d``.
* ``ModifiedDeepLabV3_DeepLoRA`` - DeepLabV3 (ResNet-101) with LoRA in layer4
                                   and a 1-channel sigmoid head (stage 1).
* ``BinaryMapTransformer``       - lightweight Transformer that classifies the
                                   predicted binary masks as live / spoof (stage 2).
* ``PretrainedDeepLabV3``        - off-the-shelf DeepLabV3 whose "person" class
                                   is used as a mask (baseline, optional).

Note on weights: ``deeplabv3_resnet101(pretrained=True)`` (used in the original
notebook) and ``deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)``
load the same checkpoint (COCO_WITH_VOC_LABELS_V1, the only weights torchvision
provides for this model). The ``weights=`` form is used here because the
``pretrained=`` argument is deprecated.
"""

import math

import torch
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101

PERSON_CLASS_INDEX = 15  # index of "person" in the 21 Pascal VOC classes


def _deeplabv3(pretrained: bool = True):
    if pretrained:
        return deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    # Random initialization without any download (used by the unit tests).
    return deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=True)


def get_mask_creator_model(device: str, pretrained: bool = True) -> nn.Module:
    """Off-the-shelf DeepLabV3 (ResNet-101) in eval mode, used to create live masks."""
    model = _deeplabv3(pretrained)
    return model.to(device).eval()


class LoRAConv2d(nn.Module):
    """
    LoRA wrapper for a Conv2d layer.

    The original convolution is frozen. A trainable low-rank branch
    ``lora_B(lora_A(x))`` is added, scaled by ``alpha / r``. ``lora_A`` keeps the
    kernel size, stride, padding, dilation and groups of the original layer and
    maps to ``r`` channels; ``lora_B`` is a 1x1 convolution back to the original
    number of output channels. ``lora_B`` is initialized to zero, so the wrapped
    layer starts exactly as the original layer.
    """

    def __init__(self, original_conv: nn.Conv2d, r: int = 32, alpha: int = 64):
        super().__init__()

        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if self.original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Conv2d(
            in_channels=original_conv.in_channels,
            out_channels=r,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=original_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_conv(x) + (self.lora_B(self.lora_A(x)) * self.scaling)


class ModifiedDeepLabV3_DeepLoRA(nn.Module):
    """
    DeepLabV3 (ResNet-101) adapted for binary live-face segmentation.

    1. Load the pre-trained DeepLabV3 and remove the auxiliary classifier.
    2. Freeze all pre-trained parameters.
    3. Replace the last classifier convolution with a new trainable
       ``Conv2d(256, 1, kernel_size=1)`` (one output channel).
    4. Wrap every ``conv2`` (3x3) convolution of the backbone's ``layer4`` with
       ``LoRAConv2d``.
    5. Apply a sigmoid to the output.

    Trainable parameters: the LoRA branches and the new 1x1 output convolution.
    """

    def __init__(self, r: int = 32, alpha: int = 64, pretrained: bool = True):
        super().__init__()

        self.model = _deeplabv3(pretrained)
        self.model.aux_classifier = None

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1)

        self._inject_lora(self.model.backbone.layer4, r=r, alpha=alpha)

        self.sigmoid = nn.Sigmoid()

    def _inject_lora(self, module: nn.Module, r: int, alpha: int) -> None:
        """Recursively replace the 'conv2' layers with LoRAConv2d."""
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and name == "conv2":
                setattr(module, name, LoRAConv2d(child, r=r, alpha=alpha))
            else:
                self._inject_lora(child, r, alpha)

    def print_trainable_parameters(self) -> None:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,}")

    def forward(self, x):
        x = self.model(x)["out"]
        return self.sigmoid(x)


class BinaryMapTransformer(nn.Module):
    """
    Lightweight Transformer classifier for 1-channel binary masks.

    Patch embedding (Conv2d with kernel = stride = patch_size) -> learned
    positional embedding -> pre-norm Transformer encoder -> global average
    pooling over patches -> LayerNorm -> MLP head (embed_dim -> mlp_dim -> 128
    -> num_classes, GELU and dropout).
    """

    def __init__(
        self,
        image_size: int = 520,
        patch_size: int = 20,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x)          # (N, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)     # (N, num_patches, D)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)                    # global average pooling
        x = self.norm(x)
        return self.classifier(x)            # logits


class PretrainedDeepLabV3(nn.Module):
    """Off-the-shelf DeepLabV3; pixels predicted as "person" form the mask."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.pretrained_model = _deeplabv3(pretrained)

    def forward(self, x):
        x = self.pretrained_model(x)["out"]
        x = torch.argmax(x, dim=1).unsqueeze(1)
        return (x == PERSON_CLASS_INDEX).float()


def count_params(model: nn.Module):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
