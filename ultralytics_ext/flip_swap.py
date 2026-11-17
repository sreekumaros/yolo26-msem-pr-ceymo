"""Direction-aware horizontal flip for road markings.

Ultralytics mirrors training images (RandomFlip with fliplr, and CopyPaste in its
default "flip" mode) but keeps every class label unchanged. For CeyMo that turns a
Left Arrow into a right-pointing arrow still labelled LA, and likewise for
Straight-Left / Straight-Right. This module patches both transforms so that a
mirrored instance gets the mirrored class (LA <-> RA, SLA <-> SRA).

The semantic target of Segment26 (`sem_masks`) is built later from `cls`, so it
follows automatically.

Call `enable_flip_class_swap(names)` in the training process before the dataloader
is built. Dataloader workers inherit the patch only with the "fork" start method
(the Linux default, also on Colab); `enable_flip_class_swap` refuses otherwise.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

# Pairs of class names that turn into each other under a left-right mirror.
CEYMO_FLIP_PAIRS = (("LA", "RA"), ("SLA", "SRA"))

_LUT: np.ndarray | None = None


def _swap(cls: np.ndarray) -> np.ndarray:
    """Map class ids through the flip lookup table, keeping shape and dtype."""
    if _LUT is None or cls is None or len(cls) == 0:
        return cls
    idx = cls.astype(np.int64)
    return _LUT[idx].astype(cls.dtype).reshape(cls.shape)


def build_lut(names, pairs=CEYMO_FLIP_PAIRS) -> np.ndarray:
    """Lookup table class_id -> class_id after a horizontal flip."""
    names = list(names.values()) if isinstance(names, dict) else list(names)
    lut = np.arange(len(names), dtype=np.int64)
    for a, b in pairs:
        if a not in names or b not in names:
            raise ValueError(f"flip pair ({a}, {b}) not found in dataset names {names}")
        ia, ib = names.index(a), names.index(b)
        lut[ia], lut[ib] = ib, ia
    return lut


def enable_flip_class_swap(names, pairs=CEYMO_FLIP_PAIRS, workers: int = 0) -> np.ndarray:
    """Patch RandomFlip and CopyPaste so mirrored instances get the mirrored class."""
    global _LUT
    from ultralytics.data import augment as A

    if workers > 0 and mp.get_context().get_start_method() != "fork":
        # Python 3.14 defaults to forkserver on Linux and spawn on macOS; those workers would miss the patch.
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("flip class swap needs 'fork' for dataloader workers on this OS; use --workers 0")
        mp.set_start_method("fork", force=True)
    for cls_name in ("RandomFlip", "CopyPaste"):
        if not hasattr(getattr(A, cls_name), "apply_instances"):
            raise RuntimeError(f"Ultralytics {cls_name} has no apply_instances(); this patch does not support this version")

    _LUT = build_lut(names, pairs)
    if getattr(A.RandomFlip, "_flip_swap_patched", False):
        return _LUT

    orig_flip = A.RandomFlip.apply_instances

    def flip_apply_instances(self, labels, params):
        labels = orig_flip(self, labels, params)
        if params.get("flip") and params.get("direction") == "horizontal" and "cls" in labels:
            labels["cls"] = _swap(labels["cls"])
        return labels

    orig_cp = A.CopyPaste.apply_instances

    def cp_apply_instances(self, labels, params=None):
        # "flip" mode pastes a mirrored copy of the image's own instances, labelled from `cls`.
        if params is not None and params.get("labels2_cls") is None:
            params = dict(params, labels2_cls=_swap(labels["cls"]))
        return orig_cp(self, labels, params)

    A.RandomFlip.apply_instances = flip_apply_instances
    A.CopyPaste.apply_instances = cp_apply_instances
    A.RandomFlip._flip_swap_patched = True
    return _LUT


def self_test(names) -> None:
    """Check the patch on a synthetic sample; raises AssertionError if labels are not swapped."""
    from ultralytics.data.augment import CopyPaste, RandomFlip
    from ultralytics.utils.instance import Instances

    lut = build_lut(names)
    nc = len(lut)

    def sample():
        n = nc
        segs = np.stack([np.array([[10 + 40 * i, 10], [30 + 40 * i, 10], [30 + 40 * i, 30], [10 + 40 * i, 30]], np.float32)
                         for i in range(n)])
        boxes = np.stack([s.min(0).tolist() + s.max(0).tolist() for s in segs]).astype(np.float32)
        inst = Instances(boxes, segments=segs, bbox_format="xyxy", normalized=False)
        # Instances sit in the left half, so CopyPaste's mirrored copies land on free space and get pasted.
        return {"img": np.zeros((64, 2 * (40 * n + 20), 3), np.uint8), "instances": inst,
                "cls": np.arange(n, dtype=np.float32).reshape(-1, 1)}

    out = RandomFlip(p=1.0, direction="horizontal")(sample())
    assert np.array_equal(out["cls"].ravel().astype(int), lut), f"RandomFlip did not swap classes: {out['cls'].ravel()}"
    out = RandomFlip(p=0.0, direction="horizontal")(sample())
    assert np.array_equal(out["cls"].ravel().astype(int), np.arange(nc)), "RandomFlip swapped without flipping"

    cp = CopyPaste(p=1.0, mode="flip")
    lab = sample()
    params = cp.get_params(lab)
    out = cp.apply_instances(lab, params)
    pasted = out["cls"].ravel().astype(int)[nc:]
    assert len(params["selected"]) == nc, "CopyPaste self-test pasted nothing; test layout is wrong"
    assert np.array_equal(pasted, lut[params["selected"]]), f"CopyPaste did not swap pasted classes: {pasted}"
