#!/usr/bin/env python3
"""Plot validation Mask mAP@0.50 and mAP@0.50:0.95 per epoch from an Ultralytics results.csv."""
from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(src: str, dst: str):
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(src))]
    ep = [int(float(r["epoch"])) for r in rows]
    m50 = [100 * float(r["metrics/mAP50(M)"]) for r in rows]
    m95 = [100 * float(r["metrics/mAP50-95(M)"]) for r in rows]
    best = max(range(len(m95)), key=m95.__getitem__)
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    ax.plot(ep, m50, lw=1.2, label="Mask mAP@0.50")
    ax.plot(ep, m95, lw=1.2, label="Mask mAP@0.50:0.95")
    ax.axvline(ep[best], color="0.5", lw=0.6, ls=":")
    ax.annotate(f"best {m95[best]:.1f} (ep. {ep[best]})", (ep[best], m95[best]), fontsize=6,
                xytext=(-60, -18), textcoords="offset points", arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel("Validation (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(dst)
    print(f"Saved {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
