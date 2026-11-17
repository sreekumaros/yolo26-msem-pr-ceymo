#!/usr/bin/env python3
"""Idempotently add the custom heads to Ultralytics' parse_model head branch.

This automates docs/MANUAL_PATCH.md: custom Segment-like heads must receive
[reg_max, end2end, [ch_P3, ch_P4, ch_P5]] exactly like Segment26. The names are
resolved with globals().get(...) so un-registered runs (e.g. the plain baseline)
are unaffected. A backup tasks.py.bak is written on first patch.

Usage:  python patch_ultralytics.py            # patch the installed ultralytics
        python patch_ultralytics.py --check    # report status only
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

MARK = "# MSEM-PR patch"
NAMES = ("MSEMPRSegment26", "MSEMOnlySegment26", "PROnlySegment26", "PlainSegment26")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    tasks = Path(importlib.util.find_spec("ultralytics.nn.tasks").origin)
    src = tasks.read_text()
    if MARK in src:
        print(f"Already patched: {tasks}")
        return
    if args.check:
        raise SystemExit(f"Not patched: {tasks}")

    anchor = "                Segment26,\n                YOLOESegment,\n"
    if src.count(anchor) != 1:
        raise SystemExit("Could not locate the Segment26 entry of parse_model's head frozenset; "
                         "apply docs/MANUAL_PATCH.md by hand for this Ultralytics version.")
    extra = "".join(f'                globals().get("{n}"),  {MARK}\n' for n in NAMES)
    shutil.copy2(tasks, tasks.with_suffix(".py.bak"))
    tasks.write_text(src.replace(anchor, "                Segment26,\n" + extra + "                YOLOESegment,\n"))
    print(f"Patched {tasks} (backup: {tasks.with_suffix('.py.bak')})")


if __name__ == "__main__":
    main()
