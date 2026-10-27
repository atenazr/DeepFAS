
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
    history = engine.fine_tune_model(model, None, optimizer, _FakeLoader(), None, cfg, "cpu", tmp_path)
    saved = sorted(p.name for p in tmp_path.glob("*.pth.tar"))
    return history, calls["train"], lrs, saved


def test_all_epochs_are_trained_without_early_stopping(monkeypatch, tmp_path):
    # IoU stops improving after epoch 3: training must still run all 20 epochs.
    ious = [0.80, 0.85, 0.90] + [0.50] * 17
    losses = [1.0] * 20
    history, n_train, _, saved = _run_schedule(monkeypatch, tmp_path, ious, losses)

    assert n_train == 20
    assert len(history) == 20
    assert history["val_iou"].tolist() == ious
    assert saved == ["Epoch-20.pth.tar"]      # only the final-epoch model is saved


def test_final_checkpoint_holds_final_model(monkeypatch, tmp_path):
    _run_schedule(monkeypatch, tmp_path, [0.5] * 3, [1.0] * 3, epochs=3)
    checkpoint = torch.load(tmp_path / "Epoch-3.pth.tar")
    assert set(checkpoint) == {"state_dict", "optimizer"}


def test_lr_decay_rules(monkeypatch, tmp_path):
    ious = [0.94, 0.951, 0.95, 0.955, 0.96, 0.9549] + [0.97 + 0.001 * i for i in range(14)]
    losses = [1.0] * 20
    history, n_train, lrs, _ = _run_schedule(monkeypatch, tmp_path, ious, losses)
    base = 1e-4
    expected = [base,                 # 0.94   -> no change
                base * 0.5,           # 0.951  -> x0.5
                base * 0.5,           # 0.95   -> no change (strict bounds)
                base * 0.5,           # 0.955  -> no change (strict bounds)
                base * 0.05,          # 0.96   -> x0.1
                base * 0.025]         # 0.9549 -> x0.5
    assert np.allclose(lrs[:6], expected)
    assert n_train == 20 and len(lrs) == 20


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


class _PersonModel(nn.Module):
    """Stand-in for the pre-trained DeepLabV3: predicts "person" in the image centre."""

    def forward(self, x):
        out = torch.zeros(x.shape[0], 21, x.shape[2], x.shape[3])
        out[:, 0] = 1.0
        out[:, 15, 2:6, 2:6] = 2.0
        return {"out": out}


def _mask_batch(tmp_path):
    images = torch.zeros(2, 3, 8, 8)
    labels = torch.tensor([1, 0])                                # live, spoof
    paths = [str(tmp_path / "live" / "mask.png"), str(tmp_path / "spoof" / "mask.png")]
    return [(images, labels, ["a", "b"], paths, ["", ""])], paths


def _white_pixels(path):
    return int((np.array(Image.open(path).convert("L")) > 0).sum())


def test_create_masks_training_targets(tmp_path):
    loader, (live_path, spoof_path) = _mask_batch(tmp_path)
    engine.create_masks(loader, _PersonModel(), "cpu", image_size=8)
    assert _white_pixels(live_path) == 16                        # pre-trained DeepLabV3 foreground
    assert _white_pixels(spoof_path) == 0                        # all black


def test_extract_embeddings(tmp_path):
    from src.models import BinaryMapTransformer

    paths = []
    for k in range(5):
        p = tmp_path / f"map_{k}.png"
        Image.new("1", (40, 40), k % 2).save(p)
        paths.append(str(p))
    loader = [(None, torch.tensor([0, 1, 0]), None, None, paths[:3]),
              (None, torch.tensor([1, 0]), None, None, paths[3:])]

    model = BinaryMapTransformer(image_size=40, patch_size=20)
    embeddings, labels = engine.extract_embeddings(model, loader, "cpu")
    assert embeddings.shape == (5, 128)                          # input of the final Linear(128, 2)
    assert labels.tolist() == [0, 1, 0, 1, 0]

    # The hook must capture exactly the input of the classification layer.
    model.eval()
    maps = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("L")) > 0).float().unsqueeze(0)
                        for p in paths])
    with torch.no_grad():
        logits = model(maps)
    final = model.classifier[-1]
    assert torch.allclose(final(torch.from_numpy(embeddings)), logits, atol=1e-6)
