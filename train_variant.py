#!/usr/bin/env python3
"""Train one ablation variant with the exact recipe used for the paper.

Hyper-parameters are identical to train_msem_pr.py of the original submission
(AdamW, lr0=1e-3, 150 epochs, box=10, cls=1.5, dfl=0, ...). Every variant is
initialised from the COCO-pretrained yolo26s-seg.pt for all layers whose shapes
match; MSEM and PR-Head layers start from random initialisation.

--flip controls horizontal mirroring (RandomFlip and CopyPaste "flip" mode):
  swap  (default) mirror and relabel direction-dependent classes (LA<->RA, SLA<->SRA)
  off   no mirroring at all (fliplr=0, CopyPaste pastes from other images instead)
  plain the original submission's behaviour: mirror but keep labels (label noise)
The chosen flip mode and image size are saved in runs/<name>/recipe.json and reused on resume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ultralytics import YOLO  # noqa: E402
from ultralytics_ext import register_msem_pr  # noqa: E402
from ultralytics_ext import flip_swap  # noqa: E402

RECIPE = dict(
    epochs=150, imgsz=640, optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=5, patience=50, amp=True, box=10.0, cls=1.5, dfl=0.0, label_smoothing=0.05,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15.0, shear=5.0, translate=0.1, scale=0.5, fliplr=0.5,
    mosaic=1.0, close_mosaic=20, mixup=0.15, copy_paste=0.3, erasing=0.4, mask_ratio=4, overlap_mask=True,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["baseline", "baseline_p256", "msem_only", "pr_only", "full"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="yolo26s-seg.pt", help="COCO-pretrained weights; 'none' trains from scratch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default=str(HERE / "runs"))
    ap.add_argument("--epochs", type=int, default=150, help="Keep 150 for the paper; smaller only for smoke tests")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size (e.g. 1280 for thin markings)")
    ap.add_argument("--flip", default="swap", choices=["swap", "off", "plain"],
                    help="Horizontal mirroring: swap LA/RA and SLA/SRA labels (default), off, or plain (original, buggy)")
    args = ap.parse_args()

    register_msem_pr()
    name = f"{args.variant}_s{args.seed}"
    run_dir = Path(args.project) / name
    done, last = run_dir / "DONE", run_dir / "weights" / "last.pt"
    if done.exists():
        print(f"{name}: already finished, skipping")
        return

    recipe_file = run_dir / "recipe.json"
    recipe = {"flip": args.flip, "imgsz": args.imgsz}
    if recipe_file.exists():
        saved = json.loads(recipe_file.read_text())
        if saved != recipe:
            print(f"{name}: using the run's saved recipe {saved} (command line asked for {recipe})")
        recipe = saved
    elif last.exists():
        # Run started before recipe.json existed: it used the original recipe.
        recipe = {"flip": "plain", "imgsz": 640}
        print(f"{name}: older run without recipe.json, resuming with the original recipe {recipe}")
    if recipe["flip"] == "swap":
        names = yaml.safe_load(Path(args.data).read_text())["names"]
        lut = flip_swap.enable_flip_class_swap(names, workers=args.workers)
        flip_swap.self_test(names)
        print(f"{name}: flip class swap on (class map after mirroring: {lut.tolist()})")

    if last.exists() and checkpoint_epoch(last) >= 0:
        # Interrupted run: continue from the last epoch with the original arguments and optimizer state.
        print(f"{name}: resuming from {last}")
        fix_empty_scaler(last)
        YOLO(str(last)).train(resume=True, device=args.device, batch=args.batch, workers=args.workers)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        recipe_file.write_text(json.dumps(recipe))
        hyp = dict(RECIPE, imgsz=recipe["imgsz"], epochs=args.epochs)
        if recipe["flip"] == "off":
            hyp.update(fliplr=0.0, copy_paste_mode="mixup")  # "mixup" mode pastes from another image, unmirrored
        else:
            hyp.update(copy_paste_mode="flip")
        model = YOLO(str(HERE / "configs" / f"yolo26s-seg-{args.variant}.yaml"))
        if args.init.lower() != "none":
            model.load(args.init)  # Ultralytics logs "Transferred N/M items from pretrained weights"
        model.train(data=args.data, seed=args.seed, deterministic=True, batch=args.batch, device=args.device,
                    workers=args.workers, project=args.project, name=name, exist_ok=True,
                    plots=True, **hyp)
    done.touch()


def checkpoint_epoch(path: Path) -> int:
    """Epoch stored in a checkpoint; Ultralytics sets it to -1 once training has finished."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1))


def fix_empty_scaler(path: Path) -> None:
    """Give a checkpoint saved with AMP off (e.g. on CPU) a fresh GradScaler state.

    Without this, resuming with AMP on fails with "The source state dict is empty".
    The original file is kept as last_noscaler.pt.
    """
    import shutil

    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "scaler" in ckpt and not ckpt["scaler"]:
        backup = path.with_name("last_noscaler.pt")
        if not backup.exists():
            shutil.copy2(path, backup)
        ckpt["scaler"] = {"scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                          "growth_interval": 2000, "_growth_tracker": 0}
        torch.save(ckpt, path)
        print(f"{path}: empty AMP scaler state replaced with defaults (backup: {backup.name})")


if __name__ == "__main__":
    main()
