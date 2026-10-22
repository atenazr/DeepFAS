import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make "src" importable when pytest is run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def fake_dataset(tmp_path):
    """
    Small dataset tree with the expected naming:
    <root>/<a>/<b>/<dataset>_<split>_<class>_<id>/cropped_<k>.jpg
    """
    root = tmp_path / "replay"
    folders = {
        "replay_train_live_client001": 5,
        "replay_train_live_client002": 3,
        "replay_train_attack_print_client001": 4,
        "replay_test_live_client003": 2,
        "replay_test_attack_video_client003": 6,
        "replay_devel_live_client004": 2,   # neither train nor test -> ignored
    }
    rng = np.random.default_rng(0)
    for folder, n in folders.items():
        d = root / "faces" / "frames" / folder
        d.mkdir(parents=True)
        for k in range(n):
            arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"cropped_{k}.jpg")
    return root, folders
