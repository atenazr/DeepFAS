"""
Run the full face anti-spoofing pipeline.

    python run_pipeline.py --config configs/default_config.yaml

Steps (all run by default, in this order):
    masks     create the training masks with the pre-trained DeepLabV3
              (live: foreground mask, spoof: all-black mask)
    finetune  LoRA fine-tuning of DeepLabV3 on these masks (20 epochs) + segmentation metrics
    maps      binary maps predicted by the fine-tuned DeepLabV3
    classify  train the lightweight Transformer on the maps, evaluate it, plot its embeddings

The sampled train/test image lists are written to <output_root>/splits/ and
reused by later runs with the same output folder, so the steps can also be run
separately (e.g. ``--steps maps classify --segmentation-checkpoint ...``).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import load_config, save_config
from src.data import build_loaders, load_file_list, sample_splits, save_file_list
from src.engine import (calculate_pixel_auc, compute_dataset_iou, create_feature_maps, create_masks,
                        evaluate_classifier, evaluate_pretrained_baseline, extract_embeddings,
                        final_checkpoint_name, fine_tune_model, plot_segmentation_examples, train_classifier)
from src.models import BinaryMapTransformer, ModifiedDeepLabV3_DeepLoRA, count_params, get_mask_creator_model
from src.transforms import get_image_transform
from src.utils import (get_device, load_checkpoint, plot_dimensionality_reduction, plot_history,
                       resolve_num_workers, save_json, set_seed)

ALL_STEPS = ["masks", "finetune", "maps", "classify"]
CLASSIFIER_CHECKPOINT = "transformer_classifier.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default_config.yaml", help="YAML config file")
    parser.add_argument("--data", nargs="+", default=None,
                        help="dataset root(s); overrides data.base_paths in the config")
    parser.add_argument("--output", default=None, help="output folder; overrides output_root")
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS, help="steps to run")
    parser.add_argument("--segmentation-checkpoint", default=None,
                        help="fine-tuned DeepLabV3 checkpoint to use when 'finetune' is not run")
    parser.add_argument("--resample", action="store_true",
                        help="sample new train/test lists even if <output>/splits/ exists")
    parser.add_argument("--seed", type=int, default=None, help="random seed; overrides the config")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'; overrides the config")
    return parser.parse_args()


def get_splits(cfg, output_root: Path, resample: bool):
    split_dir = output_root / "splits"
    train_list, test_list = split_dir / "train_files.txt", split_dir / "test_files.txt"

    if train_list.exists() and test_list.exists() and not resample:
        print(f"Using the image lists in {split_dir}")
        return load_file_list(train_list), load_file_list(test_list)

    train_files, test_files = sample_splits(cfg.data)
    save_file_list(train_files, train_list)
    save_file_list(test_files, test_list)
    return train_files, test_files


def build_segmentation_model(cfg, device):
    model = ModifiedDeepLabV3_DeepLoRA(r=cfg.segmentation.lora_rank, alpha=cfg.segmentation.lora_alpha)
    return model.to(device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data:
        cfg.data.base_paths = args.data
    if args.output:
        cfg.output_root = args.output
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device:
        cfg.device = args.device

    steps = [s for s in ALL_STEPS if s in args.steps]
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"

    save_config(cfg, output_root / "config_used.yaml")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Device: {device} | steps: {', '.join(steps)}")

    # ----------------------------------------------------------------- data
    train_files, test_files = get_splits(cfg, output_root, args.resample)
    print(f"Train images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")
    if not train_files or not test_files:
        sys.exit("No images found. Check data.base_paths and the folder naming described in README.md.")

    train_loader, test_loader = build_loaders(
        train_files, test_files, output_root,
        transform=get_image_transform(cfg.data.image_size),
        batch_size=cfg.loader.batch_size,
        num_workers=resolve_num_workers(cfg.loader.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    results = {}
    timings = {}

    # ---------------------------------------------------------- 1. masks
    if "masks" in steps:
        t0 = time.time()
        mask_model = get_mask_creator_model(device)
        create_masks(train_loader, mask_model, device, cfg.data.image_size)
        create_masks(test_loader, mask_model, device, cfg.data.image_size)
        del mask_model
        torch.cuda.empty_cache()
        timings["masks_s"] = time.time() - t0

    # ------------------------------------------------------- 2. finetune
    segmentation_model = None
    if "finetune" in steps:
        t0 = time.time()
        segmentation_model = build_segmentation_model(cfg, device)
        segmentation_model.print_trainable_parameters()
        try:
            import torchinfo
            torchinfo.summary(segmentation_model, input_size=(1, 3, cfg.data.image_size, cfg.data.image_size),
                              device=device)
        except ImportError:
            pass

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(segmentation_model.parameters(), lr=cfg.segmentation.learning_rate)

        # Trains for cfg.segmentation.epochs epochs and saves checkpoints/Epoch-<epochs>.pth.tar.
        history = fine_tune_model(segmentation_model, loss_function, optimizer,
                                  train_loader, test_loader, cfg.segmentation, device, checkpoint_dir)
        plot_history(history, output_root)
        timings["finetune_s"] = time.time() - t0

        # Segmentation metrics of the final-epoch model.
        test_iou = compute_dataset_iou(segmentation_model, test_loader, device)
        print(f"Test IoU (fine-tuned DeepLabV3): {test_iou:.4f}")
        segmentation_results = {"test_iou": test_iou, "epochs": cfg.segmentation.epochs,
                                "trainable_params": count_params(segmentation_model)[1]}
        if cfg.evaluation.plot_examples:
            plot_segmentation_examples(segmentation_model, test_loader, device, test_iou,
                                       output_root / "images-masks-finetuned")
        if cfg.evaluation.pixel_auc:
            segmentation_results["pixel_auc"] = calculate_pixel_auc(segmentation_model, test_loader,
                                                                    device, output_root)
        if cfg.evaluation.pretrained_baseline:
            segmentation_results["pretrained_test_iou"] = evaluate_pretrained_baseline(test_loader, device)
        save_json(segmentation_results, output_root / "segmentation_metrics.json")
        results["segmentation"] = segmentation_results

    # ----------------------------------------------------------- 3. maps
    if "maps" in steps:
        t0 = time.time()
        if segmentation_model is None:
            checkpoint = (args.segmentation_checkpoint
                          or checkpoint_dir / final_checkpoint_name(cfg.segmentation.epochs))
            if not Path(checkpoint).exists():
                sys.exit(f"Segmentation checkpoint not found: {checkpoint}")
            segmentation_model = build_segmentation_model(cfg, device)
            load_checkpoint(checkpoint, segmentation_model, device=device)
        create_feature_maps(segmentation_model, test_loader, device)
        create_feature_maps(segmentation_model, train_loader, device)
        timings["maps_s"] = time.time() - t0

    del segmentation_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------- 4. classify
    if "classify" in steps:
        t0 = time.time()
        c = cfg.classifier
        classifier = BinaryMapTransformer(
            image_size=cfg.data.image_size, patch_size=c.patch_size, in_channels=1,
            embed_dim=c.embed_dim, num_heads=c.num_heads, num_layers=c.num_layers,
            mlp_dim=c.mlp_dim, dropout=c.dropout, num_classes=c.num_classes,
        ).to(device)
        total, trainable = count_params(classifier)
        print(f"Classifier parameters: {total:,} (trainable {trainable:,})")

        classifier_history = train_classifier(classifier, train_loader, c, device)
        classifier_history.to_csv(output_root / "classifier_training.csv", index=False)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), checkpoint_dir / CLASSIFIER_CHECKPOINT)
        timings["classifier_train_s"] = time.time() - t0

        metrics = evaluate_classifier(classifier, test_loader, device, output_root, cfg.evaluation.threshold)
        metrics["params"] = total
        results["classifier"] = metrics
        write_metrics(output_root, results, timings)   # keep the metrics even if a plot below fails

        if cfg.evaluation.embedding_plots:
            embeddings, labels = extract_embeddings(classifier, test_loader, device)
            metrics.update(plot_dimensionality_reduction(embeddings, labels, output_root))

    write_metrics(output_root, results, timings)
    print(f"Results written to {output_root.resolve()}")


def write_metrics(output_root: Path, results: dict, timings: dict) -> None:
    """Merge with the results of earlier runs in the same output folder (when steps are run separately)."""
    metrics_path = output_root / "metrics.json"
    merged = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    merged.update(results)
    merged["timings"] = {**merged.get("timings", {}), **timings}
    save_json(merged, metrics_path)


if __name__ == "__main__":
    main()
