from pathlib import Path

import torch
from PIL import Image

from src.data import (FaceAntiSpoofingDataset, collect_images, label_from_folder, load_binary_masks,
                      load_file_list, output_dir_for, round_robin_sample, save_file_list)
from src.transforms import get_image_transform


def test_collect_images_uses_official_partitions(fake_dataset):
    root, folders = fake_dataset
    train = collect_images([str(root)], "train")
    test = collect_images([str(root)], "test")

    assert set(train) == {f for f in folders if "_train_" in f}
    assert set(test) == {f for f in folders if "_test_" in f}
    for groups in (train, test):
        for folder, paths in groups.items():
            assert len(paths) == folders[folder]


def test_collect_images_cross_dataset(tmp_path, fake_dataset):
    root, folders = fake_dataset
    other = tmp_path / "other" / "x" / "y" / "msu_test_live_client009"
    other.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(other / "cropped_0.jpg")

    # Last root = test set (all of its folders), the others = training set (all folders).
    train = collect_images([str(root), str(tmp_path / "other")], "train")
    test = collect_images([str(root), str(tmp_path / "other")], "test")
    assert set(train) == set(folders)
    assert set(test) == {"msu_test_live_client009"}


def test_round_robin_sample_is_balanced(fake_dataset):
    root, _ = fake_dataset
    groups = collect_images([str(root)], "train")   # folders with 5, 3, 4 images

    sampled = round_robin_sample(groups, 9)
    assert len(sampled) == 9
    assert len(set(sampled)) == 9
    per_folder = {f: sum(p.parent.name == f for p in sampled) for f in groups}
    assert sorted(per_folder.values()) == [3, 3, 3]

    # Asking for more than available returns everything once.
    assert len(round_robin_sample(groups, 100)) == 12


def test_label_and_output_paths(tmp_path):
    assert label_from_folder("replay_train_live_client001") == 1
    assert label_from_folder("replay_train_attack_print_client001") == 0

    img = Path("/data/replay/faces/frames/replay_train_live_client001/cropped_3.jpg")
    assert output_dir_for(img, tmp_path) == tmp_path / "faces" / "frames" / "replay_train_live_client001"


def test_dataset_item(fake_dataset, tmp_path):
    root, _ = fake_dataset
    files = sorted(root.rglob("cropped_*.jpg"))
    ds = FaceAntiSpoofingDataset(files, tmp_path / "out", get_image_transform(40))

    image, label, img_path, mask_path, map_path = ds[0]
    assert image.shape == (3, 40, 40)
    assert label.dtype == torch.long
    assert Path(mask_path).name == f"mask_{Path(img_path).stem}.png"
    assert Path(map_path).name == f"map_{Path(img_path).stem}.png"
    assert Path(mask_path).parent == output_dir_for(Path(img_path), tmp_path / "out")


def test_file_list_roundtrip(tmp_path):
    files = [Path("/a/b/c_train_live_1/cropped_1.jpg"), Path("/a/b/c_train_live_1/cropped_2.jpg")]
    save_file_list(files, tmp_path / "splits" / "train_files.txt")
    assert load_file_list(tmp_path / "splits" / "train_files.txt") == files


def test_load_binary_masks(tmp_path):
    white = tmp_path / "w.png"
    black = tmp_path / "b.png"
    Image.new("1", (10, 10), 1).save(white)
    Image.new("1", (10, 10), 0).save(black)

    masks = load_binary_masks([str(white), str(black)])
    assert masks.shape == (2, 1, 10, 10)
    assert masks.dtype == torch.int64
    assert masks[0].sum() == 100 and masks[1].sum() == 0
