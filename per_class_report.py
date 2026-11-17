#!/usr/bin/env python3
"""Print a compact summary of evaluate.py results: one row per variant/split (mean over seeds).

    python per_class_report.py results_flipswap_640 [results_old ...]

Shows Mask mAP, mIoU, Boundary IoU and per-class IoU, so the effect of a recipe change
(e.g. the flip fix on LA/RA/SLA/SRA) is visible at a glance. Several folders are printed one after another.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(folder: Path) -> None:
    groups = defaultdict(list)
    for f in sorted(folder.glob("*_s*_*.json")):
        m = re.match(r"(.+)_s(\d+)_(val|test)\.json$", f.name)
        if m:
            groups[(m.group(1), m.group(3))].append(json.loads(f.read_text()))
    if not groups:
        print(f"{folder}: no results")
        return
    classes = list(next(iter(groups.values()))[0]["IoU_per_class"])
    head = f"{'variant':15}{'split':6}{'n':>2} {'mAP50':>6}{'mAP':>6}{'mIoU':>6}{'BIoU3':>6}{'FPS':>6} | " + "".join(f"{c:>5}" for c in classes)
    print(f"\n== {folder}\n{head}\n{'-' * len(head)}")
    for (v, split), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        mean = lambda get: 100 * np.mean([get(r) for r in rs])  # noqa: E731
        fps = next((r["speed"]["fps"] for r in rs if r.get("speed", {}).get("fps")), None)
        row = (f"{v:15}{split:6}{len(rs):>2} {mean(lambda r: r['mask']['mAP50']):6.1f}{mean(lambda r: r['mask']['mAP50_95']):6.1f}"
               f"{mean(lambda r: r['mIoU']):6.1f}{mean(lambda r: r['BIoU_3px']):6.1f}{(f'{fps:.0f}' if fps else '-'):>6} | ")
        row += "".join(f"{mean(lambda r, c=c: r['IoU_per_class'][c]):5.1f}" for c in classes)
        print(row)


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["results"]:
        summarize(Path(arg))
