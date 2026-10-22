"""
Training and evaluation steps of the pipeline.

Step 1  create_masks            training masks: live -> DeepLabV3 foreground, spoof -> all black
Step 2  fine_tune_model         LoRA fine-tuning of DeepLabV3 on these masks
Step 3  create_feature_maps     binary maps predicted by the fine-tuned DeepLabV3
Step 4  train_classifier        lightweight Transformer on the binary maps
Step 5  evaluate_classifier     Accuracy, FAR/FRR/HTER, AUC, APCER/BPCER/ACER
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .data import LIVE_LABEL, SPOOF_LABEL, load_binary_masks
from .models import PretrainedDeepLabV3
from .transforms import binarize
from .utils import calculate_iou, plot_images_and_masks, plot_roc, save_checkpoint, save_json


# ===========================================================================
# Step 1 - training masks
# ===========================================================================
MASK_THRESHOLD = 40  # gray-level threshold applied to the palette image


def save_live_mask(output_prediction: torch.Tensor, path: str, image_size: int) -> None:
    """
    Turn the 21-class DeepLabV3 prediction into a black/white mask.

    Each class index is mapped to a palette color, the colored image is
    converted to gray levels, and pixels brighter than ``MASK_THRESHOLD`` become
    white. The background class (index 0, black) therefore stays black and the
    predicted foreground classes become white.
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(output_prediction.byte().cpu().numpy()).resize((image_size, image_size))
    r.putpalette(colors)

    image_file = r.convert("L")
    image_file = image_file.point(lambda p: 255 if p > MASK_THRESHOLD else 0)
    img = image_file.convert("1")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_spoof_mask(path: str, image_size: int) -> None:
    """All-black mask for attack samples."""
    spoofed_mask = Image.new(mode="1", size=(image_size, image_size))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    spoofed_mask.save(path)


def create_live_mask(image: torch.Tensor, mask_path: str, mask_model, device: str, image_size: int) -> None:
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = mask_model(image)["out"][0]
    output_prediction = output.argmax(0)
    save_live_mask(output_prediction, mask_path, image_size)


def create_masks(loader, mask_model, device: str, image_size: int, spoofed_masks: bool = False) -> None:
    """
    Write the training mask of every image in ``loader``.

    Default: live samples get the foreground mask of the pre-trained DeepLabV3,
    spoof samples get an all-black mask.
    ``spoofed_masks=True``: only spoof samples are processed and they receive
    the pre-trained DeepLabV3 mask (live samples are skipped).
    """
    for images, labels, _, mask_paths, _ in tqdm(loader, desc="Creating masks"):
        for image, label, mask_path in zip(images, labels, mask_paths):
            if spoofed_masks:
                if label.item() == SPOOF_LABEL:
                    create_live_mask(image, mask_path, mask_model, device, image_size)
            else:
                if label.item() == LIVE_LABEL:
                    create_live_mask(image, mask_path, mask_model, device, image_size)
                else:
                    create_spoof_mask(mask_path, image_size)


# ===========================================================================
# Step 2 - LoRA fine-tuning of DeepLabV3
# ===========================================================================
def train_segmentation_epoch(model, loader, loss_function, optimizer, device: str) -> float:
    """
    Train for one epoch and return the accumulated loss.

    The loss of each batch (batch mean) is accumulated as
    ``loss.item() * images[0].size(0)``, exactly as in the experiments of the
    paper. ``images[0].size(0)`` is the number of image channels (3), so the
    logged training loss is 3 x the sum of batch-mean losses; this scaling only
    affects the logged value and the checkpoint-saving rule, not the gradients.
    """
    model.train()
    running_loss = 0.0

    train_loop = tqdm(loader, leave=True)
    for images, _, _, mask_paths, _ in train_loop:
        images = images.to(device)
        masks = load_binary_masks(mask_paths).float().to(device)

        outputs = model(images)
        loss = loss_function(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images[0].size(0)
        train_loop.set_postfix(loss=loss.item())

    return running_loss


def compute_dataset_iou(model, loader, device: str) -> float:
    """
    Mean IoU over the batches of ``loader`` (IoU computed per batch over all of
    its pixels, then averaged over batches), rounded to 4 decimals. Model
    outputs are binarized at 0.5.
    """
    model.eval()
    iou_list = []
    with torch.no_grad():
        for images, _, _, mask_paths, _ in loader:
            images = images.to(device)
            masks = load_binary_masks(mask_paths).to(device)

            outputs = binarize(model(images))
            iou_list.append(calculate_iou(outputs, masks))

    return round(sum(iou_list) / len(iou_list), 4)


def plot_segmentation_examples(model, loader, device: str, mean_iou, save_path) -> None:
    """Plot samples 10-17 of the first batch: image, mask, prediction, overlay."""
    model.eval()
    with torch.no_grad():
        images, _, _, mask_paths, _ = next(iter(loader))
        start, end = (10, 18) if len(images) > 10 else (0, min(8, len(images)))
        masks = load_binary_masks(mask_paths[start:end]).to(device)
        images = images[start:end].to(device)
        outputs = binarize(model(images))
    plot_images_and_masks(images, masks, outputs, mean_iou, save_path=save_path)


def apply_lr_decay(optimizer, epoch_iou: float, cfg) -> None:
    """
    Learning-rate decay driven by the epoch IoU (applied after every epoch):
    ``lower < IoU < upper``  -> lr *= factor_mid   (0.95 < IoU < 0.955 -> x0.5)
    ``IoU > upper``          -> lr *= factor_high  (IoU > 0.955        -> x0.1)
    """
    if cfg.lr_decay_lower < epoch_iou < cfg.lr_decay_upper:
        factor = cfg.lr_decay_factor_mid
    elif epoch_iou > cfg.lr_decay_upper:
        factor = cfg.lr_decay_factor_high
    else:
        return
    for param_group in optimizer.param_groups:
        param_group["lr"] *= factor


def fine_tune_model(model, loss_function, optimizer, train_loader, val_loader, cfg, device: str,
                    checkpoint_dir):
    """
    Fine-tune the LoRA-adapted DeepLabV3.

    After each epoch the IoU on ``val_loader`` is computed (in the paper's
    experiments this is the test partition; there is no separate validation
    split). It is used for:

    * Early stopping, active from epoch index ``early_stopping_start_epoch``
      (10, i.e. the 11th epoch). An epoch "improves" when its IoU is strictly
      higher than every previous epoch's IoU. A non-improving epoch increases a
      counter; an improving one resets it. When a non-improving epoch finds the
      counter already above ``early_stopping_patience`` (3), training stops,
      i.e. on the 5th consecutive non-improving epoch. That last epoch is
      trained but not added to the history.
    * Checkpointing (same epochs): ``Epoch-<n>.pth.tar`` is saved when the IoU
      improves or the training loss is lower than every previous epoch's loss.
    * Learning-rate decay, see ``apply_lr_decay``.

    The model returned is the model after the last trained epoch; no
    checkpoint is reloaded.

    Returns ``(history, info)``: per-epoch ``train_loss`` / ``val_iou`` and a
    dict with the number of trained epochs and whether early stopping fired.
    """
    checkpoint_dir = Path(checkpoint_dir)
    train_loss_history = []
    val_iou_history = []

    counter = 0
    epochs_trained = 0
    stopped_early = False

    for epoch in range(cfg.epochs):
        print(f"Epoch: {epoch + 1}/{cfg.epochs} " + "-" * 62)

        running_loss = train_segmentation_epoch(model, train_loader, loss_function, optimizer, device)
        epoch_loss = running_loss / len(train_loader.dataset)
        epochs_trained = epoch + 1

        epoch_val_iou = compute_dataset_iou(model, val_loader, device)

        if epoch >= cfg.early_stopping_start_epoch:
            iou_improved = all(i < epoch_val_iou for i in val_iou_history)
            loss_improved = all(i > epoch_loss for i in train_loss_history)

            if cfg.save_epoch_checkpoints and (iou_improved or loss_improved):
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                save_checkpoint(checkpoint, checkpoint_dir / f"Epoch-{epoch + 1}.pth.tar")

            if iou_improved:
                counter = 0
            elif counter > cfg.early_stopping_patience:
                print(f"Train Loss: {epoch_loss:.4f} - Validation IoU: {epoch_val_iou:.4f}")
                print("No improvement in validation IoU. Stopping training.")
                stopped_early = True
                break
            else:
                counter += 1

        val_iou_history.append(epoch_val_iou)
        train_loss_history.append(epoch_loss)

        print(f"Train Loss: {epoch_loss:.4f} - Validation IoU: {epoch_val_iou:.4f}")

        apply_lr_decay(optimizer, epoch_val_iou, cfg)

    history = pd.DataFrame({"train_loss": train_loss_history, "val_iou": val_iou_history})
    print("Training is Done.")
    return history, {"epochs_trained": epochs_trained, "stopped_early": stopped_early}


def calculate_pixel_auc(model, loader, device: str, output_root) -> float:
    """Pixel-level ROC-AUC of the sigmoid output against the training masks."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, _, _, mask_paths, _ in tqdm(loader, desc="Pixel AUC"):
            images = images.to(device)
            masks = load_binary_masks(mask_paths).to(device)

            outputs = model(images)            # sigmoid is part of the model
            masks = (masks > 0.5).float()

            all_preds.append(outputs.cpu().view(-1))
            all_targets.append(masks.cpu().view(-1))

    y_scores = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy().astype(int)

    if len(np.unique(y_true)) < 2:
        print("Cannot compute AUC or plot ROC - only one class in y_true.")
        return float("nan")

    auc = roc_auc_score(y_true, y_scores)
    print(f"Pixel AUC (fine-tuned DeepLabV3): {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plot_roc(fpr, tpr, auc, "Fine-tuned DeepLabV3", "ROC Curve",
             Path(output_root) / "finetuned-deeplabv3-roc.png")
    return float(auc)


def evaluate_pretrained_baseline(loader, device: str) -> float:
    """IoU of the off-the-shelf DeepLabV3 ("person" class) against the training masks."""
    model = PretrainedDeepLabV3().to(device)
    return compute_dataset_iou(model, loader, device)


# ===========================================================================
# Step 3 - binary maps from the fine-tuned DeepLabV3
# ===========================================================================
def create_feature_maps(model, loader, device: str) -> None:
    """Save the binarized prediction of the fine-tuned DeepLabV3 for every image."""
    model.eval()
    with torch.no_grad():
        for images, _, _, _, map_paths in tqdm(loader, desc="Creating binary maps"):
            outputs = binarize(model(images.to(device)))
            for i in range(len(images)):
                path = Path(map_paths[i])
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(outputs[i].detach().cpu(), str(path))


# ===========================================================================
# Step 4 - classifier
# ===========================================================================
def train_classifier(model, train_loader, cfg, device: str) -> pd.DataFrame:
    """Train the classifier on the binary maps with cross-entropy and AdamW."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for _, labels, _, _, map_paths in train_loader:
            maps = load_binary_masks(map_paths).to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits = model(maps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100.0 * correct / total
        average_loss = running_loss / len(train_loader)
        history.append({"epoch": epoch + 1, "loss": average_loss, "accuracy": train_accuracy})
        print(f"Epoch [{epoch + 1}/{cfg.epochs}] Loss: {average_loss:.4f} Accuracy: {train_accuracy:.2f}%")

    return pd.DataFrame(history)


# ===========================================================================
# Step 5 - evaluation
# ===========================================================================
def predict_classifier(model, loader, device: str):
    """Return labels, argmax predictions and P(live) for every sample in ``loader``."""
    model.eval()
    all_labels, all_predictions, all_probabilities = [], [], []

    with torch.no_grad():
        for _, labels, _, _, map_paths in loader:
            maps = load_binary_masks(map_paths).to(device).float()
            logits = model(maps)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def compute_classification_metrics(labels, predictions, probabilities, threshold: float = 0.5) -> dict:
    """
    Live = 1 (bona fide), spoof = 0 (attack). Metrics are computed per sample.

    * Accuracy, FAR, FRR, HTER use the argmax prediction.
        FAR = spoof accepted as live / all spoof,  FRR = live rejected / all live,
        HTER = (FAR + FRR) / 2
    * AUC uses P(live).
    * APCER, BPCER, ACER use ``P(live) >= threshold``.
        APCER = attacks classified as live / all attacks,
        BPCER = bona fide classified as attack / all bona fide,
        ACER = (APCER + BPCER) / 2
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    accuracy = 100.0 * np.sum(predictions == labels) / len(labels)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[SPOOF_LABEL, LIVE_LABEL]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    hter = (far + frr) / 2

    auc = roc_auc_score(labels, probabilities)

    threshold_predictions = (probabilities >= threshold).astype(int)
    attack_samples = labels == SPOOF_LABEL
    bona_fide_samples = labels == LIVE_LABEL
    apcer = np.sum((threshold_predictions == LIVE_LABEL) & attack_samples) / np.sum(attack_samples)
    bpcer = np.sum((threshold_predictions == SPOOF_LABEL) & bona_fide_samples) / np.sum(bona_fide_samples)
    acer = (apcer + bpcer) / 2

    return {
        "accuracy": float(accuracy),
        "FAR": float(far), "FRR": float(frr), "HTER": float(hter),
        "AUC": float(auc),
        "APCER": float(apcer), "BPCER": float(bpcer), "ACER": float(acer),
        "threshold": threshold,
        "num_samples": int(len(labels)),
        "num_live": int(bona_fide_samples.sum()),
        "num_spoof": int(attack_samples.sum()),
    }


def evaluate_classifier(model, loader, device: str, output_root, threshold: float = 0.5) -> dict:
    labels, predictions, probabilities = predict_classifier(model, loader, device)
    metrics = compute_classification_metrics(labels, predictions, probabilities, threshold)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    plot_roc(fpr, tpr, metrics["AUC"], "Transformer", "ROC Curve - Lightweight Transformer",
             Path(output_root) / "transformer_roc_curve.png", dpi=300)

    save_json(metrics, Path(output_root) / "classifier_metrics.json")

    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"FAR  : {metrics['FAR']:.4f}   FRR  : {metrics['FRR']:.4f}   HTER : {metrics['HTER']:.4f}")
    print(f"AUC  : {metrics['AUC']:.4f}")
    print(f"APCER: {metrics['APCER']:.4f}   BPCER: {metrics['BPCER']:.4f}   ACER : {metrics['ACER']:.4f}")
    return metrics
