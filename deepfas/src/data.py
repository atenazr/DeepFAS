"""
Data loading.

Expected layout
---------------
Each dataset root contains one folder per video, holding the cropped face
frames of that video as ``cropped_*.jpg``. The folder name encodes the
partition and the class, separated by underscores::

    <dataset>_<split>_<class>_...        e.g.  replay_train_live_real_client001_...

* ``<split>``  is ``train`` or ``test`` (official partitions of the dataset).
* ``<class>``  is ``live`` for bona fide samples; any other value is an attack.

The video folders can sit at any depth below the root. The masks and maps
produced by the pipeline are written to
``<output_root>/<grandparent of grandparent>/<grandparent>/<video folder>/``,
mirroring the last three levels of the input path.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

LIVE_LABEL = 1
SPOOF_LABEL = 0


def collect_images(roots: Sequence[str], split: str, pattern: str = "cropped_*.jpg") -> Dict[str, List[Path]]:
    """
    Group image paths by their video folder.

    With one root, only folders whose name contains ``_<split>_`` are kept
    (official train/test partitions). With several roots, the last root is the
    test set and the other roots form the training set; in that case all
    folders of the selected roots are used.

    The images inside each folder are shuffled.
    """
    groups = defaultdict(list)
    roots = list(roots)

    if len(roots) == 1:
        root = Path(roots[0])
        for img_path in root.rglob(pattern):
            folder = img_path.parent.name
            if f"_{split}_" in folder:
                groups[folder].append(img_path)
    else:
        selected = [roots[-1]] if split == "test" else roots[:-1]
        for root in selected:
            for img_path in Path(root).rglob(pattern):
                folder = img_path.parent.name
                groups[folder].append(img_path)

    for folder in groups:
        random.shuffle(groups[folder])

    return groups


def round_robin_sample(groups: Dict[str, List[Path]], target_size: int) -> List[Path]:
    """
    Take one image from each folder in turn (folder order reshuffled every
    round) until ``target_size`` images are selected or all folders are
    exhausted. This spreads the sample evenly over the videos.
    """
    folders = list(groups.keys())
    sampled = []
    indices = {folder: 0 for folder in folders}

    while len(sampled) < target_size:
        progress = False
        random.shuffle(folders)

        for folder in folders:
            idx = indices[folder]
            if idx < len(groups[folder]):
                sampled.append(groups[folder][idx])
                indices[folder] += 1
                progress = True
                if len(sampled) == target_size:
                    break

        if not progress:  # every folder is exhausted
            break

    return sampled


def sample_splits(data_cfg):
    """Sample the train and test image lists as described in the paper."""
    n_test = int(round(data_cfg.max_number_data * data_cfg.train_test_split))
    n_train = data_cfg.max_number_data - n_test

    train_groups = collect_images(data_cfg.base_paths, "train", data_cfg.image_pattern)
    test_groups = collect_images(data_cfg.base_paths, "test", data_cfg.image_pattern)

    train_files = round_robin_sample(train_groups, n_train)
    test_files = round_robin_sample(test_groups, n_test)
    return train_files, test_files


def save_file_list(files: Sequence[Path], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")


def load_file_list(path: Path) -> List[Path]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Path(line) for line in lines if line.strip()]


def label_from_folder(folder_name: str) -> int:
    """``<dataset>_<split>_<class>_...``: 1 if <class> is 'live', else 0."""
    parts = folder_name.split("_")
    return LIVE_LABEL if parts[2] == "live" else SPOOF_LABEL


def output_dir_for(img_path: Path, output_root: Path) -> Path:
    """Folder under ``output_root`` where the mask and map of an image are stored."""
    return (Path(output_root)
            / img_path.parent.parent.parent.name
            / img_path.parent.parent.name
            / img_path.parent.name)


class FaceAntiSpoofingDataset(Dataset):
    """
    Returns ``(image, label, image_path, mask_path, map_path)``.

    ``mask_path`` is the training mask of the image (created in the mask
    creation step); ``map_path`` is the binary map predicted by the fine-tuned
    DeepLabV3 (created after fine-tuning, input of the classifier).
    """

    def __init__(self, files: Sequence[Path], output_root: str, transform=None):
        self.files = [Path(f) for f in files]
        self.output_root = Path(output_root)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(label_from_folder(img_path.parent.name), dtype=torch.long)

        out_dir = output_dir_for(img_path, self.output_root)
        mask_path = out_dir / f"mask_{img_path.stem}.png"
        map_path = out_dir / f"map_{img_path.stem}.png"

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path), str(mask_path), str(map_path)


def load_binary_masks(paths: Sequence[str]) -> torch.Tensor:
    """Read mask/map PNGs as a (N, 1, H, W) int64 tensor with values {0, 1} (pixel > 0 -> 1)."""
    return torch.stack([
        torch.from_numpy(
            (np.array(Image.open(p).convert("L")) > 0).astype(np.int64)
        ).unsqueeze(0)
        for p in paths
    ]).long()


def build_loaders(train_files, test_files, output_root, transform, batch_size, num_workers, pin_memory):
    train_dataset = FaceAntiSpoofingDataset(train_files, output_root, transform)
    test_dataset = FaceAntiSpoofingDataset(test_files, output_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
