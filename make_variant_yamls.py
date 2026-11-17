#!/usr/bin/env python3
"""Write the five ablation model YAMLs from the base yolo26s-seg.yaml.

baseline  : Segment26                      (bare YOLO26s-seg)
baseline_p256 : PlainSegment26             (bare head, prototype width matched to the variants below)
msem_only : MSEMOnlySegment26              (MSEM, no PR-Head)
pr_only   : PROnlySegment26                (PR-Head, no MSEM)
full      : MSEMPRSegment26                (MSEM + PR-Head, the proposed model)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HEADS = {
    "baseline": None,
    "baseline_p256": "PlainSegment26",
    "msem_only": "MSEMOnlySegment26",
    "pr_only": "PROnlySegment26",
    "full": "MSEMPRSegment26",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).parent / "configs" / "yolo26s-seg.yaml"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "configs"))
    ap.add_argument("--nc", type=int, default=11)
    ap.add_argument("--pr-k", type=int, default=196)
    ap.add_argument("--pr-iters", type=int, default=3)
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.src).read_text())
    base["nc"] = args.nc
    for name, head in HEADS.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))
        last = cfg["head"][-1]
        assert last[2] == "Segment26", f"unexpected final layer {last}"
        if head:
            last[2] = head
            last[3] = ["nc", 32, 256, args.pr_k, args.pr_iters]
        dst = Path(args.out) / f"yolo26s-seg-{name}.yaml"
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"{dst.name}: {last}")


if __name__ == "__main__":
    main()
