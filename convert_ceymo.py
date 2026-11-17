#!/usr/bin/env python3
"""Convert the official CeyMo release into a leakage-free YOLO-seg dataset.

Input (official CeyMo layout):
    <ceymo>/train/images/*.jpg, <ceymo>/train/polygon_annotations/*.json   (2,099 images)
    <ceymo>/test/images/*.jpg,  <ceymo>/test/polygon_annotations/*.json    (788 images)
Each JSON is a LabelMe file whose `shapes` are class-labelled polygons in pixel
coordinates of the 1920x1080 frame.

Output:
    <out>/{train,val,test}/images, <out>/{train,val,test}/labels, <out>/data.yaml, <out>/split_stats.json
Each label line is `cls x1 y1 x2 y2 ... xn yn` with vertices normalised to [0, 1]
(Ultralytics YOLO-seg polygon format). Ultralytics rasterises these polygons into
instance masks with cv2.fillPoly at train/val time.

Protocol:
  * test = the official CeyMo test split, unchanged.
  * val  = whole scenes held out from the official train split. A scene is the
           numeric file prefix (e.g. 120, 120a, 120b are consecutive frames of one
           scene), so no scene is shared between train and val. Among --search
           seeded scene shuffles, the one whose per-class validation share is
           closest to --val-frac is kept (class-stratified, deterministic).
  * Train scenes whose prefix also occurs in the official test split are dropped
    from train, so no scene is shared between train/val and test either.
  * No offline augmentation is applied here; all augmentation happens online
    inside the Ultralytics dataloader on training batches only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

NAMES = ["BL", "CL", "DM", "JB", "LA", "PC", "RA", "SA", "SL", "SLA", "SRA"]
CLS = {n: i for i, n in enumerate(NAMES)}


def scene(stem: str) -> str:
    return re.match(r"\d+", stem).group(0)


def yolo_lines(js: dict) -> list[str]:
    w, h = js["imageWidth"], js["imageHeight"]
    lines = []
    for s in js["shapes"]:
        if s.get("shape_type", "polygon") != "polygon" or len(s["points"]) < 3:
            raise ValueError(f"unexpected shape {s.get('shape_type')} with {len(s['points'])} points")
        xy = " ".join(f"{min(max(x / w, 0), 1):.6f} {min(max(y / h, 0), 1):.6f}" for x, y in s["points"])
        lines.append(f"{CLS[s['label']]} {xy}")
    return lines


def write_split(items, split, out: Path, symlink: bool, stats):
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for img, ann in items:
        js = json.loads(ann.read_text())
        lines = yolo_lines(js)
        counts.update(NAMES[int(l.split()[0])] for l in lines)
        dst = out / split / "images" / img.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(img.resolve(), dst) if symlink else shutil.copy2(img, dst)
        (out / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
    stats[split] = {
        "images": len(items),
        "scenes": len({scene(i.stem) for i, _ in items}),
        "instances": sum(counts.values()),
        "per_class": {n: counts[n] for n in NAMES},
    }


def collect(root: Path, split: str):
    items = []
    for img in sorted((root / split / "images").glob("*.jpg")):
        ann = root / split / "polygon_annotations" / f"{img.stem}.json"
        if not ann.exists():
            raise FileNotFoundError(ann)
        items.append((img, ann))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceymo", required=True, help="Root of the official CeyMo release")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction of train scenes held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search", type=int, default=2000, help="Candidate scene shuffles for class stratification")
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")
    args = ap.parse_args()

    root, out = Path(args.ceymo), Path(args.out)
    train_all, test = collect(root, "train"), collect(root, "test")

    test_scenes = {scene(i.stem) for i, _ in test}
    dropped = [it for it in train_all if scene(it[0].stem) in test_scenes]
    train_pool = [it for it in train_all if scene(it[0].stem) not in test_scenes]

    by_scene = defaultdict(list)
    for it in train_pool:
        by_scene[scene(it[0].stem)].append(it)
    scene_cls = {s: Counter(sh["label"] for _, a in v for sh in json.loads(a.read_text())["shapes"])
                 for s, v in by_scene.items()}
    total_cls = sum(scene_cls.values(), Counter())
    target = round(args.val_frac * len(train_pool))

    def draw(seed):
        scenes = sorted(by_scene, key=int)
        random.Random(seed).shuffle(scenes)
        picked, n = set(), 0
        for s in scenes:
            if n >= target:
                break
            picked.add(s)
            n += len(by_scene[s])
        got = sum((scene_cls[s] for s in picked), Counter())
        # worst per-class deviation from the requested validation fraction
        return max(abs(got[c] / total_cls[c] - args.val_frac) for c in NAMES), picked

    # Class-stratified scene hold-out: deterministic search over shuffles, keep the most balanced.
    dev, val_scenes = min((draw(args.seed * 10000 + i) for i in range(args.search)), key=lambda t: t[0])
    train = [it for it in train_pool if scene(it[0].stem) not in val_scenes]
    val = [it for it in train_pool if scene(it[0].stem) in val_scenes]

    stats = {"protocol": "official CeyMo test; scene-grouped val from official train",
             "seed": args.seed, "val_frac": args.val_frac, "max_class_fraction_deviation": round(dev, 4),
             "train_images_dropped_scene_overlap_with_test": len(dropped)}
    for split, items in (("train", train), ("val", val), ("test", test)):
        write_split(items, split, out, args.symlink, stats)

    sc = {k: {scene(i.stem) for i, _ in v} for k, v in (("train", train), ("val", val), ("test", test))}
    stats["scene_overlap"] = {"train-val": len(sc["train"] & sc["val"]),
                              "train-test": len(sc["train"] & sc["test"]),
                              "val-test": len(sc["val"] & sc["test"])}
    assert not any(stats["scene_overlap"].values()), stats["scene_overlap"]

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
