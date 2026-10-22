from pathlib import Path

import pytest

from src.config import Config, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"


def test_default_config_matches_paper_defaults():
    # The YAML file and the code defaults must describe the same experiment
    # (dataset path aside).
    cfg = load_config(str(CONFIG))
    expected = Config()
    cfg.data.base_paths = expected.data.base_paths
    assert cfg == expected


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("segmentation:\n  lora_rnak: 16\n")
    with pytest.raises(KeyError):
        load_config(str(bad))


def test_float_strings_are_accepted(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("segmentation:\n  learning_rate: 1e-4\n")
    assert load_config(str(f)).segmentation.learning_rate == pytest.approx(1e-4)
