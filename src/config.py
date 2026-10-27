"""
Experiment configuration.

All hyperparameters of the pipeline live in a YAML file (see
``configs/default_config.yaml``). The defaults below are the values used for
the experiments reported in the paper, so a missing key in the YAML file falls
back to the paper setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    # Dataset roots. With one root, the official train/test partitions are
    # read from that dataset (folder names contain "_train_" / "_test_").
    # With several roots, the last root is the test set and all others are
    # used for training (cross-dataset setting).
    base_paths: List[str] = field(default_factory=lambda: ["data/replay-attack"])
    image_pattern: str = "cropped_*.jpg"   # face crops inside each video folder
    max_number_data: int = 4200            # total number of sampled images
    train_test_split: float = 0.25         # fraction of max_number_data used for test
    image_size: int = 520                  # images are resized to image_size x image_size


@dataclass
class LoaderConfig:
    batch_size: int = 28
    num_workers: Optional[int] = None      # None -> os.cpu_count()


@dataclass
class SegmentationConfig:
    """Stage 1: LoRA fine-tuning of DeepLabV3 (ResNet-101) for binary masks."""
    epochs: int = 20                       # fixed number of epochs (no early stopping)
    learning_rate: float = 1.0e-4
    lora_rank: int = 32
    lora_alpha: int = 64
    # Learning-rate decay driven by the per-epoch IoU.
    lr_decay_lower: float = 0.95           # lower < IoU < upper  -> lr *= lr_decay_factor_mid
    lr_decay_upper: float = 0.955          # IoU > upper          -> lr *= lr_decay_factor_high
    lr_decay_factor_mid: float = 0.5
    lr_decay_factor_high: float = 0.1


@dataclass
class ClassifierConfig:
    """Stage 2: lightweight Transformer classifier on the predicted binary masks."""
    epochs: int = 15
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    patch_size: int = 20
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 2


@dataclass
class EvaluationConfig:
    threshold: float = 0.5                 # threshold on P(live) for APCER/BPCER/ACER
    pixel_auc: bool = True                 # pixel-level ROC-AUC of the fine-tuned DeepLabV3
    plot_examples: bool = True             # figure with images, masks and predictions
    pretrained_baseline: bool = False      # IoU of the off-the-shelf DeepLabV3 (person class)
    embedding_plots: bool = True           # PCA / t-SNE / UMAP of the classifier embeddings (test set)


@dataclass
class Config:
    output_root: str = "results"
    seed: Optional[int] = None             # None -> no seeding (setting of the paper's runs)
    device: Optional[str] = None           # None -> "cuda" if available, else "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict):
    """Create dataclass ``cls`` from a (possibly partial) dict, recursively."""
    values = values or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")

    kwargs = {}
    for name, f in known.items():
        if name not in values:
            continue
        default = f.default_factory() if callable(f.default_factory) else f.default
        value = values[name]
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value)
        elif isinstance(default, float) and isinstance(value, (int, str)):
            # PyYAML reads "1e-4" (without a dot) as a string; accept it anyway.
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    """Load a YAML config file. Keys that are not given keep the paper defaults."""
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return _build(Config, values)


def save_config(config: Config, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
