"""Helpers: device, seeding, checkpoints, plots, result files."""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend; figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .transforms import inv_transform  # noqa: E402


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def get_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy and PyTorch. ``None`` leaves everything unseeded."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_num_workers(num_workers: Optional[int]) -> int:
    return os.cpu_count() if num_workers is None else num_workers


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"===> Saving checkpoint: {path}")
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None, device: str = "cpu") -> None:
    print(f"===> Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def apply_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.65):
    """Overlay a binary mask on an image with the given color and transparency."""
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    masked = masked.filled()
    return cv2.addWeighted(image, 1 - alpha, masked, alpha, 0)


def calculate_iou(pred_mask: torch.Tensor, true_masks: torch.Tensor) -> float:
    """IoU of binary masks, computed over all pixels of the given tensors."""
    intersection = torch.sum(pred_mask * true_masks)
    union = torch.sum(pred_mask + true_masks) - intersection
    iou = intersection / (union + 1e-6)
    return iou.item()


def plot_images_and_masks(images, masks, predicted_masks=None, mean_iou=None, save_path=None):
    """Images, ground-truth masks, predicted masks (optional) and overlays, one column per sample."""
    num_samples = len(images)
    num_rows = 3 if predicted_masks is None else 4

    fig, axes = plt.subplots(num_rows, num_samples,
                             figsize=(num_samples * 5, num_samples + (num_rows * 2)),
                             dpi=200, facecolor="#252627", squeeze=False)

    for i in range(num_samples):
        image = inv_transform(images[i]).permute(1, 2, 0).detach().cpu().numpy()
        axes[0, i].imshow(image, cmap="gray")
        axes[0, i].set_title("Image", fontsize=20, fontweight="bold", color="white")
        axes[0, i].axis("off")

        mask = masks[i].permute(1, 2, 0).detach().cpu().numpy()
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title("Actual Mask", fontsize=20, fontweight="bold", color="white")
        axes[1, i].axis("off")

        if predicted_masks is not None:
            predicted_mask = predicted_masks[i].permute(1, 2, 0).detach().cpu().numpy()
            axes[2, i].imshow(predicted_mask, cmap="gray")
            axes[2, i].set_title(
                f"Predicted M - IoU: {round(float(calculate_iou(predicted_masks[i], masks[i])), 3)}",
                fontsize=19, fontweight="bold", color="white")
            axes[2, i].axis("off")
            overlayed_image = apply_mask_overlay(image, predicted_mask)
            overlay_row = 3
        else:
            overlayed_image = apply_mask_overlay(image, mask)
            overlay_row = 2

        # imshow clips to [0, 1] anyway; clipping here only silences the matplotlib warning.
        axes[overlay_row, i].imshow(np.clip(overlayed_image, 0, 1), cmap="gray")
        axes[overlay_row, i].set_title("Overlayed Image", fontsize=20, fontweight="bold", color="white")
        axes[overlay_row, i].axis("off")

    plt.suptitle("Images and Masks" + (f" - Mean IoU: {mean_iou}" if mean_iou is not None else ""),
                 fontsize=30, fontweight="bold", color="white")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(save_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, output_root: Path) -> None:
    """Training loss and per-epoch IoU of the DeepLabV3 fine-tuning (log scale)."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_root / "training.csv", index=False)

    fig = plt.figure()
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_iou"], label="validation IOU", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Train Loss & IoU validation over Epochs")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_root / "loss-function-train-validation-deeplabv3.pdf", bbox_inches="tight")
    plt.savefig(output_root / "loss-function-train-validation-deeplabv3.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_roc(fpr, tpr, auc, label, title, save_path: Path, dpi: int = 600) -> None:
    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dimensionality_reduction(embeddings, labels, output_root, class_names=("spoof", "live"),
                                  title_prefix="BinaryMapTransformer", random_state=42) -> dict:
    """
    2-D PCA, t-SNE and UMAP projections of the classifier embeddings, colored by
    class (index 0 = spoof, 1 = live). Saved as ``<title_prefix>_{PCA,tSNE,UMAP}.png``.

    t-SNE: perplexity 30, 1000 iterations; t-SNE and UMAP use ``random_state=42``.
    Returns the PCA explained-variance ratios.
    """
    import seaborn as sns
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    class_names = list(class_names)
    hue = [class_names[int(x)] for x in labels]
    palette = sns.color_palette("husl", len(class_names))

    def scatter(points, title, xlabel, ylabel, filename):
        fig = plt.figure(figsize=(8, 6))
        sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=hue, palette=palette, alpha=0.7)
        plt.title(f"{title} - {title_prefix} Embeddings", fontsize=14)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title="Classes")
        plt.tight_layout()
        plt.savefig(output_root / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Computing PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    print("PCA explained variance:", pca.explained_variance_ratio_)
    scatter(pca_result, "PCA", "Principal Component 1", "Principal Component 2", f"{title_prefix}_PCA.png")

    perplexity = 30
    if len(embeddings) > perplexity:
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=random_state)
        tsne_result = tsne.fit_transform(embeddings)
        scatter(tsne_result, "t-SNE", "t-SNE Dimension 1", "t-SNE Dimension 2", f"{title_prefix}_tSNE.png")
    else:
        print(f"Skipping t-SNE: it needs more than {perplexity} samples (perplexity {perplexity}).")

    print("Computing UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_result = umap_reducer.fit_transform(embeddings)
    scatter(umap_result, "UMAP", "UMAP Dimension 1", "UMAP Dimension 2", f"{title_prefix}_UMAP.png")

    return {"pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_]}
