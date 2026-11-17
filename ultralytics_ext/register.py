"""Runtime registration helpers for Ultralytics custom modules."""
from __future__ import annotations

import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import head as head_mod

from .segment_msem_pr import MSEMPRSegment26, MSEMOnlySegment26, PROnlySegment26, PlainSegment26
from .msem_pr_modules import MSEM, PRHead, BoundaryLoss


def register_msem_pr():
    """
    Register custom classes into Ultralytics parse_model namespace.

    Call before YOLO('custom.yaml') or YOLO('custom.pt').
    """
    # tasks.parse_model resolves module names from globals() in ultralytics.nn.tasks.
    tasks.MSEMPRSegment = MSEMPRSegment26
    tasks.MSEMPRSegment26 = MSEMPRSegment26
    tasks.MSEMOnlySegment26 = MSEMOnlySegment26
    tasks.PROnlySegment26 = PROnlySegment26
    tasks.PlainSegment26 = PlainSegment26
    tasks.MSEM = MSEM
    tasks.PRHead = PRHead
    tasks.BoundaryLoss = BoundaryLoss

    # Also expose in head module for checkpoint loading/import friendliness.
    head_mod.MSEMPRSegment = MSEMPRSegment26
    head_mod.MSEMPRSegment26 = MSEMPRSegment26
    head_mod.MSEMOnlySegment26 = MSEMOnlySegment26
    head_mod.PROnlySegment26 = PROnlySegment26
    head_mod.PlainSegment26 = PlainSegment26
    return MSEMPRSegment26
