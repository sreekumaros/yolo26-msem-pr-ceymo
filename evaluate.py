#!/usr/bin/env python3
"""Evaluate one trained variant on one split with a single, fixed protocol.

Reports (written to <out>/<variant>_<split>.json):
  * Ultralytics COCO-style Box/Mask P, R, mAP@0.50, mAP@0.50:0.95 (+ per-class mask AP).
  * Semantic mIoU over the 11 marking classes (background excluded from the mean),
    from a confusion matrix accumulated over the whole split at the native
    1920x1080 label resolution.
  * Boundary IoU (Cheng et al., CVPR 2021): for class c, G_d = pixels of the GT
    class-c region within d px of its contour (inner band), P_d likewise for the
    prediction; intersections/unions are accumulated over the split per class
    and averaged over classes. Reported for d = 3 px (strict) and for
    d = 2% of the image diagonal (the default of Cheng et al.).
  * Optional --fps: pure forward-pass throughput at 640x640, batch 1, FP16 on CUDA.

Predicted instance masks are painted in ascending confidence so the most
confident instance wins where masks overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics import YOLO  # noqa: E402
from ultralytics_ext import register_msem_pr  # noqa: E402


def gt_semantic(label_path: Path, h: int, w: int) -> np.ndarray:
    gt = np.zeros((h, w), np.uint8)
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            p = line.split()
            if len(p) < 7:
                continue
            pts = (np.array(p[1:], np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
            cv2.fillPoly(gt, [pts], int(p[0]) + 1)
    return gt


def pred_semantic(result, h: int, w: int) -> np.ndarray:
    pred = np.zeros((h, w), np.uint8)
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return pred
    masks = result.masks.data.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    order = np.argsort(result.boxes.conf.cpu().numpy())  # ascending: highest confidence painted last
    for i in order:
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        pred[m > 0.5] = cls[i] + 1
    return pred


def inner_band(mask: np.ndarray, d: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    m = mask.astype(np.uint8)
    # pad with background so markings touching the image border still have a contour there
    ero = cv2.erode(cv2.copyMakeBorder(m, d, d, d, d, cv2.BORDER_CONSTANT, value=0), k)[d:-d, d:-d]
    return (m > 0) & (ero == 0)


def resolve(data: dict, data_yaml: Path, split: str) -> Path:
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    p = Path(data[split])
    return p if p.is_absolute() else (root / p).resolve()


def measure_fps(weights: str, imgsz: int, iters: int = 500, warmup: int = 50) -> dict:
    if not torch.cuda.is_available():
        return {"fps": None, "note": "CUDA not available; FPS not measured"}
    # Load a fresh copy: the model used by .predict() was fused inside torch.inference_mode(),
    # so its weights are inference tensors and .half() on them raises
    # "Inference tensors do not track version counter".
    net = YOLO(weights).model
    net = net.fuse(verbose=False) if hasattr(net, "fuse") else net  # same conv+BN fusion as Ultralytics predict
    net = net.cuda().eval().half()
    x = torch.zeros(1, 3, imgsz, imgsz, device="cuda", dtype=torch.half)
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            net(x)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    return {"fps": 1.0 / dt, "latency_ms": dt * 1e3, "device": torch.cuda.get_device_name(0),
            "precision": "fp16", "batch": 1, "imgsz": imgsz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for mIoU / Boundary IoU maps")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2, help="Dataloader workers (0 if /dev/shm is small)")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only use the first N images for mIoU/BIoU")
    ap.add_argument("--fps", action="store_true")
    args = ap.parse_args()

    register_msem_pr()
    data_yaml = Path(args.data).resolve()
    data = yaml.safe_load(data_yaml.read_text())
    names, nc = data["names"], int(data["nc"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"variant": args.variant, "split": args.split, "weights": str(args.weights), "conf": args.conf}

    model = YOLO(args.weights)
    res["params_M"] = sum(p.numel() for p in model.model.parameters()) / 1e6

    # 1) COCO-style AP via Ultralytics (conf=0.001 is the standard AP protocol)
    m = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
                  conf=0.001, iou=0.7, plots=True, project=str(out / "ultra"), name=f"{args.variant}_{args.split}",
                  exist_ok=True, verbose=False)
    res["box"] = {"P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50_95": m.box.map}
    res["mask"] = {"P": m.seg.mp, "R": m.seg.mr, "mAP50": m.seg.map50, "mAP50_95": m.seg.map}
    res["mask_per_class"] = {names[c]: {"AP50": float(m.seg.ap50[i]), "AP50_95": float(m.seg.ap[i])}
                             for i, c in enumerate(m.seg.ap_class_index)}

    # 2) mIoU and Boundary IoU at native label resolution
    img_dir = resolve(data, data_yaml, args.split)
    lbl_dir = img_dir.parent / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]
    conf_mat = np.zeros((nc + 1, nc + 1), np.int64)
    widths = {"3px": None, "2pct_diag": None}
    b_int = {k: np.zeros(nc) for k in widths}
    b_uni = {k: np.zeros(nc) for k in widths}
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        widths = {"3px": 3, "2pct_diag": max(1, round(0.02 * (h * h + w * w) ** 0.5))}
        gt = gt_semantic(lbl_dir / f"{img_path.stem}.txt", h, w)
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=args.device, retina_masks=True, verbose=False)[0]
        pr = pred_semantic(r, h, w)
        conf_mat += np.bincount(gt.ravel().astype(np.int64) * (nc + 1) + pr.ravel(),
                                minlength=(nc + 1) ** 2).reshape(nc + 1, nc + 1)
        for c in range(1, nc + 1):
            g, p = gt == c, pr == c
            if not g.any() and not p.any():
                continue
            for k, d in widths.items():
                gb, pb = inner_band(g, d), inner_band(p, d)
                b_int[k][c - 1] += np.logical_and(gb, pb).sum()
                b_uni[k][c - 1] += np.logical_or(gb, pb).sum()

    tp = np.diag(conf_mat)[1:]
    union = conf_mat[1:, :].sum(1) + conf_mat[:, 1:].sum(0) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    res["mIoU"] = float(np.nanmean(iou))
    res["IoU_per_class"] = {names[i]: float(iou[i]) for i in range(nc)}
    res["boundary_widths_px"] = widths
    for k in b_int:
        biou = np.where(b_uni[k] > 0, b_int[k] / np.maximum(b_uni[k], 1), np.nan)
        res[f"BIoU_{k}"] = float(np.nanmean(biou))
        res[f"BIoU_{k}_per_class"] = {names[i]: float(biou[i]) for i in range(nc)}
    res["n_images_semantic"] = len(images)

    if args.fps:
        try:
            res["speed"] = measure_fps(args.weights, args.imgsz)
        except Exception as e:  # never lose the accuracy results because of the speed benchmark
            print(f"FPS measurement failed: {e}")
            res["speed"] = {"fps": None, "error": str(e)}

    dst = out / f"{args.variant}_{args.split}.json"
    dst.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("variant", "split", "mask", "mIoU", "BIoU_3px", "BIoU_2pct_diag")}, indent=2))
    print(f"Saved {dst}")


if __name__ == "__main__":
    main()
