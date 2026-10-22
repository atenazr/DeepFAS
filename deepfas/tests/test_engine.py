
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import src.engine as engine
from src.config import SegmentationConfig


class _FakeLoader:
    dataset = list(range(10))


def _run_schedule(monkeypatch, tmp_path, ious, losses, epochs=20):
    """Run fine_tune_model with fixed per-epoch IoU and loss values."""
    ious, losses = iter(ious), iter(losses)
    calls = {"train": 0}

    def fake_train(*args, **kwargs):
        calls["train"] += 1
        return next(losses) * len(_FakeLoader.dataset)

    monkeypatch.setattr(engine, "train_segmentation_epoch", fake_train)
    monkeypatch.setattr(engine, "compute_dataset_iou", lambda *a, **k: next(ious))

    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lrs = []
    orig_decay = engine.apply_lr_decay

    def record_decay(opt, iou, cfg):
        orig_decay(opt, iou, cfg)
        lrs.append(opt.param_groups[0]["lr"])

    monkeypatch.setattr(engine, "apply_lr_decay", record_decay)

    cfg = SegmentationConfig(epochs=epochs)
    history, info = engine.fine_tune_model(model, None, optimizer, _FakeLoader(), None, cfg, "cpu", tmp_path)
    saved = sorted(int(p.name.split("-")[1].split(".")[0]) for p in tmp_path.glob("Epoch-*.pth.tar"))
    return history, info, calls["train"], lrs, saved


def test_early_stopping_fires_on_fifth_non_improving_epoch(monkeypatch, tmp_path):
    # Improves until epoch 11 (index 10), then never again.
    ious = [0.80 + 0.01 * i for i in range(11)] + [0.85] * 9
    losses = [1.0 - 0.01 * i for i in range(20)]
    history, info, n_train, lrs, saved = _run_schedule(monkeypatch, tmp_path, ious, losses)

    assert n_train == 16                      # epochs 12-15 increase the counter, epoch 16 stops
    assert info == {"epochs_trained": 16, "stopped_early": True}
    assert len(history) == 15                 # the stopping epoch is not added to the history
    # Checkpoints from epoch 11 on, when IoU or training loss improves (loss improves every epoch here).
    assert saved == list(range(11, 17))


def test_no_early_stopping_before_epoch_11(monkeypatch, tmp_path):
    ious = [0.5] * 20
    losses = [1.0] * 20
    history, info, n_train, _, saved = _run_schedule(monkeypatch, tmp_path, ious, losses)
    # Epochs 1-10 are never checked; epochs 11-14 count up, epoch 15 stops.
    assert n_train == 15 and info["stopped_early"] and len(history) == 14
    assert saved == []                        # nothing improves strictly


def test_lr_decay_rules(monkeypatch, tmp_path):
    ious = [0.94, 0.951, 0.95, 0.955, 0.96, 0.9549] + [0.97 + 0.001 * i for i in range(14)]
    losses = [1.0] * 20
    _, info, _, lrs, _ = _run_schedule(monkeypatch, tmp_path, ious, losses)
    base = 1e-4
    expected = [base,                 # 0.94   -> no change
                base * 0.5,           # 0.951  -> x0.5
                base * 0.5,           # 0.95   -> no change (strict bounds)
                base * 0.5,           # 0.955  -> no change (strict bounds)
                base * 0.05,          # 0.96   -> x0.1
                base * 0.025]         # 0.9549 -> x0.5
    assert np.allclose(lrs[:6], expected)
    assert info == {"epochs_trained": 20, "stopped_early": False}


def test_classification_metrics():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.4, 0.7, 0.1, 0.6, 0.2, 0.3])   # 1 live rejected, 1 spoof accepted
    preds = (probs > 0.5).astype(int)
    m = engine.compute_classification_metrics(labels, preds, probs, threshold=0.5)

    assert m["FAR"] == pytest.approx(0.25) and m["FRR"] == pytest.approx(0.25)
    assert m["HTER"] == pytest.approx(0.25)
    assert m["APCER"] == pytest.approx(0.25) and m["BPCER"] == pytest.approx(0.25)
    assert m["ACER"] == pytest.approx(m["HTER"])
    assert m["accuracy"] == pytest.approx(75.0)
    assert m["AUC"] == pytest.approx(15 / 16)                   # only (0.4 live, 0.6 spoof) is mis-ranked


def test_live_and_spoof_masks(tmp_path):
    prediction = torch.zeros(8, 8, dtype=torch.long)
    prediction[2:6, 2:6] = 15                                   # "person"
    engine.save_live_mask(prediction, str(tmp_path / "live.png"), image_size=8)
    engine.create_spoof_mask(str(tmp_path / "spoof.png"), image_size=8)

    live = np.array(Image.open(tmp_path / "live.png").convert("L")) > 0
    spoof = np.array(Image.open(tmp_path / "spoof.png").convert("L")) > 0
    assert live.sum() == 16 and live[2:6, 2:6].all()
    assert spoof.sum() == 0
